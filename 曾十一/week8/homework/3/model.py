import torch
import torch.nn as nn

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout, 
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        embedded = self.embedding(token_seq)
        outputs, hidden = self.rnn(embedded)
        hidden_state = hidden[0] + hidden[1]  # 双向GRU的两个方向加和
        return hidden_state, outputs

# 注意力机制
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim)
        self.dec_proj = nn.Linear(dec_dim, attn_dim)

    def forward(self, enc_output, dec_output):
        # enc_output: [batch_size, src_len, enc_dim]
        # dec_output: [batch_size, tgt_len, dec_dim]
        enc_proj = self.enc_proj(enc_output)
        dec_proj = self.dec_proj(dec_output)

        scores = torch.bmm(enc_proj, dec_proj.permute(0, 2, 1))  # [batch, src_len, tgt_len]
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.permute(0, 2, 1), enc_output)  # [batch, tgt_len, enc_dim]
        return context

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout,
                          batch_first=True)
        self.attention = Attention(enc_dim=hidden_dim*2, dec_dim=hidden_dim, attn_dim=hidden_dim)
        self.attention_fc = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, token_seq, hidden_state, enc_output):
        embedded = self.embedding(token_seq)
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))

        context = self.attention(enc_output, dec_output)
        combined = torch.cat((dec_output, context), dim=-1)
        out = torch.tanh(self.attention_fc(combined))
        logits = self.fc(out)
        return logits

# Seq2Seq整体封装
class Seq2Seq(nn.Module):
    def __init__(self, enc_emb_size, dec_emb_size, emb_dim, hidden_size, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout)
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout)

    def forward(self, enc_input, dec_input):
        encoder_hidden, enc_output = self.encoder(enc_input)
        logits = self.decoder(dec_input, encoder_hidden, enc_output)
        return logits


if __name__ == '__main__':
    
    # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10

    # encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # hidden_state = encoder(token_seq)  # Encoder输出（最后时间步状态）
    # # print(hidden_state.shape)  # 应该是 [batch_size, hidden_dim]

    # # 测试Decoder
    # decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # logits = decoder(token_seq, hidden_state)  # Decoder输出
    # print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )

    logits,_ = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

