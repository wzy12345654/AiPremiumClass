import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.d_k = hidden_size // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def _attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        x, attn = self._attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, -1)
        return self.out(x)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))

        if memory_mask is not None:
            dec_len = x.size(1)
            enc_len = memory.size(1)
            
            if dec_len > enc_len:
                padding = torch.zeros((memory_mask.size(0), dec_len - enc_len, enc_len), 
                                    dtype=torch.bool, device=memory_mask.device)
                memory_mask = torch.cat([memory_mask, padding], dim=1)
            else:
                memory_mask = memory_mask[:, :dec_len, :]

        cross_attn = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        ff = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff))


class MyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["hidden_size"] % cfg["n_heads"] == 0
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.encoder_pos = PositionalEncoding(cfg["hidden_size"], cfg["encoder_max_length"])
        self.decoder_pos = PositionalEncoding(cfg["hidden_size"], cfg["decoder_max_length"])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(cfg["hidden_size"], cfg["n_heads"], cfg["ff_size"], cfg["dropout"])
            for _ in range(cfg["encoder_layers"])
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(cfg["hidden_size"], cfg["n_heads"], cfg["ff_size"], cfg["dropout"])
            for _ in range(cfg["decoder_layers"])
        ])
        self.fc_out = nn.Linear(cfg["hidden_size"], cfg["vocab_size"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.scale = math.sqrt(cfg["hidden_size"])
        self.encoder_max_len = cfg["encoder_max_length"]
        self.decoder_max_len = cfg["decoder_max_length"]

    def generate_encoder_mask(self, mask):
        if mask is None:
            return None
        if mask.dim() == 3:
            return mask
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        return mask.unsqueeze(2) & mask.unsqueeze(1)

    def generate_decoder_mask(self, mask):
        if mask is None:
            return None
        if mask.dim() == 3:
            return mask
        
        _, seq_len = mask.size()
        assert seq_len <= self.decoder_max_len

        if mask.dtype != torch.bool:
            mask = mask > 0.5
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device))
        seq_mask = mask.unsqueeze(1)
        block_mask = seq_mask & seq_mask.transpose(1,2)
        
        return (causal_mask & block_mask).unsqueeze(1)
    
    def encode(self, src, src_mask=None):
        assert src.size(1) <= self.encoder_max_len
        src = self.embedding(src) * self.scale
        src += self.encoder_pos(src)
        src = self.dropout(src)
        src_mask = self.generate_encoder_mask(src_mask) if src_mask is not None else None
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        assert tgt.size(1) <= self.decoder_max_len
        tgt = self.embedding(tgt) * self.scale
        tgt += self.decoder_pos(tgt)
        tgt = self.dropout(tgt)
        tgt_mask = self.generate_decoder_mask(tgt_mask) if tgt_mask is not None else None
        memory_mask = self.generate_encoder_mask(memory_mask) if memory_mask is not None else None
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.fc_out(tgt)

    def forward(self, x_encoder, x_decoder, encoder_mask=None, decoder_mask=None):
        memory = self.encode(x_encoder, encoder_mask)
        output = self.decode(x_decoder, memory, decoder_mask, encoder_mask)
        return {'out': output.permute(0, 2, 1)}
