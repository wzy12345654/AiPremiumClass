import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_seq_len=5000):
        super().__init__()

        # 构建位置编码矩阵 [max_seq_len, embedding_dim]
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(- torch.arange(0, embedding_dim, 2) * math.log(10000) / embedding_dim)

        # 偶数位：sin(position / (10000^(2i/dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位：cos(position / (10000^(2i/dim)))
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度：[1, max_seq_len, embedding_dim]
        pe = pe.unsqueeze(0)

        self.drop = nn.Dropout(dropout)

        # 注册为 buffer，不作为参数参与训练，但保存到模型中
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        # 加 positional encoding（不参与训练）
        x = x + self.pe[:, :seq_len, :].detach()
        return self.drop(x)


class PoetryTransformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, num_heads, num_enc_layers, num_dec_layers, dropout, max_seq_len):
        super().__init__()

        self.transformer = nn.Transformer(d_model=embedding_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=embedding_dim * 2,
                                          dropout=dropout,
                                          batch_first=True)

        self.enc_emb = nn.Embedding(enc_vocab_size, embedding_dim)
        self.dec_emb = nn.Embedding(dec_vocab_size, embedding_dim)
        self.predict = nn.Linear(embedding_dim, dec_vocab_size)
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, dropout)

    def forward(self, enc_inpout, dec_inpout, tgt_mask=None, enc_padding_mask=None, dec_padding_mask=None):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inpout))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inpout))

        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_padding_mask,
                         tgt_key_padding_mask=dec_padding_mask)

        return self.predict(outs)

    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)