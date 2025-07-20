import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# 位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 位置编码索引 (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)
        # 添加和batch对应维度 (1, 5000, emb_size)
        pos_embdding = pos_embdding.unsqueeze(-2)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)  # token长度
        # (1, token_len, emb_size)
        add_emb = self.pos_embedding[:token_len, :] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        # transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder input embedding
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        # decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # enc_inp: (batch_size, enc_len) batch_size: 一批语料数量 enc_len: 编码器输入长度
        # dec_inp: (batch_size, dec_len) batch_size: 一批语料数量 dec_len: 解码器输入长度
        # tgt_mask: (dec_len, dec_len) 解码器上三角填充-inf的mask
        # enc_pad_mask: (batch_size, enc_len) 编码器填充0的mask
        # dec_pad_mask: (batch_size, dec_len) 解码器填充0的mask
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 推理
        return self.predict(outs)
    
    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)
 
if __name__ == '__main__':
    
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    print(dec_tokens)


