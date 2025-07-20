import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset
import torch.nn.functional as F
import math


# 位置编码方法
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size , dropout, max_len=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(- torch.arange(0 , emb_size , 2) * math.log(10000) / emb_size)
        # 位置编码索引
        pos = torch.arange(0 , max_len) . reshape(max_len , 1)
        # 编码矩阵
        pos_embedding = torch.zeros((max_len , emb_size))
        pos_embedding[:,0::2] = torch.sin(pos * den)
        pos_embedding[:,1::2] = torch.cos(pos * den)
        # 添加batch维度
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding: [batch_size, seq_len, embedding_dim]
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self , d_model , nhead , num_enc_layers , num_dec_layers , dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        # transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder input embedding  编码
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        # decoder input embedding  解码
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear  线性
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src = enc_emb , tgt = dec_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=enc_pad_mask,
                                tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)

    # 编码器
    def encode(self , enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        memory = self.transformer.encoder(enc_emb)
        return memory

    # 解码器  需要传入mask掩码
    def decode(self, dec_inp, memory, tgt_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        outs = self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask)
        return outs

if __name__ == '__main__':
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= " 君不见，黄河之水天上来，奔流到海不复回。君不见，高堂明镜悲白发，朝如青丝暮成雪。 "
    chs = list(corpus)

    enc_tokens, dec_tokens = [],[]













