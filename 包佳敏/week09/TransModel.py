import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

#位置编码矩阵
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #位置编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros(max_len, emb_size)
        #位置编码索引（5000，1）
        position = torch.arange(0, max_len, dtype=torch.float).reshape(max_len, 1)  # (max_len, 1)
        #行缩放指数值
        den = torch.exp(torch.arange(0, emb_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / emb_size))
        pos_embdding[:, 0::2] = torch.sin(position * den) #奇数列
        pos_embdding[:, 1::2] = torch.cos(position * den) #偶数列
        #添加和batch对应纬度（1，5000， emb_size）
        pos_embdding = pos_embdding.unsqueeze(0)  # (1, max_len, d_model)
        #dropput
        self.dropout = nn.Dropout(dropout)
        #注册当前矩阵不参与梯度更新,注册到缓冲区
        self.register_buffer('pos_embdding', pos_embdding)

    #前向传播,让系统回调
    def forward(self, token_embdding):
        #token_embdding: (batch_size, seq_len, emb_size)    
        token_len = token_embdding.size(1) #token长度
        #pos_embdding: (1, token_len, emb_size)
        add_emb= self.pos_embdding[:token_len, :] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTrans(nn.Module):
    def __init__(self, input_dim, output_dim, emb_size, n_heads, num_layers, dropout=0.1):
        super(Seq2SeqTrans, self).__init__()
        # input_dim: 输入词汇表大小
        # output_dim: 输出词汇表大小
        # emb_size: 词向量维度
        self.enc_embedding = nn.Embedding(input_dim, emb_size)
        self.dec_embedding = nn.Embedding(output_dim, emb_size)
        #token预测基于解码器词典
        self.predict = nn.Linear(emb_size, output_dim)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        # Transformer模型 d_model词向量大小，nhead头的数量，num_encoder_layers和num_decoder_layers分别表示编码器和解码器的堆叠层数
        self.transformer = nn.Transformer(d_model=emb_size, nhead=n_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward = 4 * emb_size,
                                          dropout=dropout,batch_first=True)

    def forward(self, enc_inp, dec_inp, tgt_mask,tgt_key_padding_mask,src_key_padding_mask):
        #multi head attention之前基于位置编码embedding生成
        enc_emb = self.positional_encoding(self.enc_embedding(enc_inp))
        dec_emb = self.positional_encoding(self.dec_embedding(dec_inp))
        #调用transformer计算
        output = self.transformer(enc_emb, dec_emb, tgt_mask, 
                                  src_key_padding_mask=tgt_key_padding_mask,
                                  tgt_key_padding_mask=src_key_padding_mask)
        #推理
        return self.predict(output)  
    
    #推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.positional_encoding(self.enc_embedding(enc_inp))
        return self.transformer.encoder(enc_emb)
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.positional_encoding(self.dec_embedding(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)       