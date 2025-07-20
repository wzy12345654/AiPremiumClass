import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import math


# 构建位置编码的矩阵
class PositionnalEmbedding(nn.Module):
    def __init__(self,emb_size,dropout,maxLen=5000):
        super().__init__()
        # 行缩放指数
        den = torch.exp(- torch.arange(0,emb_size,2) * math.log(10000)/emb_size)
        # 位置编码索引
        pos = torch.arange(0,maxLen).reshape(maxLen,1)
        # 构建位置编码矩阵 (5000,emb_size)
        pos_embedding = torch.zeros(maxLen,emb_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den) # 偶数位置
        pos_embedding[:, 1::2] = torch.cos(pos * den) # 奇数位置
        # 给 pos_embedding 添加一个维度，变成 (1,5000,emb_size)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)
        # 注册为模型的一个buffer，不会被优化器更新
        self.register_buffer('pos_embedding',pos_embedding)
        
    def forward(self, token_embedding):
      # token_embedding 的形状是 (batch_size,seq_len,emb_size) 
      return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
    
      
      
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model,nhead,num_enc_layers,num_dec_layers,dim_forward,dropout,
                 enc_voc_size,dec_voc_size):
        super().__init__()
        # 构建transformer的位置编码矩阵
        self.transformer = nn.Transformer(
                                 d_model=d_model,
                                 nhead=nhead,
                                 num_encoder_layers=num_enc_layers,
                                 num_decoder_layers=num_dec_layers,
                                 dim_feedforward=dim_forward,
                                 dropout=dropout,
                                 batch_first=True
                                 ) 
        # 构建 encoder input embedding矩阵，输入的是token_index，输出是token_index对应的embedding向量
        self.enc_embedding = nn.Embedding(enc_voc_size,d_model)
        # 构建 decoder input embedding矩阵，输入的是token_index，输出是token_index对应的embedding向量
        self.dec_embedding = nn.Embedding(dec_voc_size,d_model)
        # 构建 predict generate linear层，输入是decoder的输出，输出是decoder的预测结果
        self.predict = nn.Linear(d_model,dec_voc_size) # token预测基础解码器词典
        # 构建位置编码矩阵 pos_encoding，输入是token_index，输出是token_index对应的位置编码向量
        self.pos_encoding = PositionnalEmbedding(d_model,dropout)

    def forward(self,enc_input,dec_input,tgt_mask,enc_pad_mask,dec_pad_mask):
      
       # mutihead attention 之前基于位置编码embedding 声场
       enc_embedding = self.pos_encoding(self.enc_embedding(enc_input)) # (batch_size,seq_len,emb_size)
       dec_embedding = self.pos_encoding(self.dec_embedding(dec_input)) # (batch_size,seq_len,emb_size)
       # 调用 transformer 计算
       output = self.transformer(
                              src=enc_embedding, # encoder 的输入   
                              tgt=dec_embedding, # decoder 的输入    
                              tgt_mask = tgt_mask, # decoder 的掩码矩阵
                              src_key_padding_mask=enc_pad_mask, # encoder 的填充掩码
                              tgt_key_padding_mask=dec_pad_mask, # decoder 的填充掩码
                            )
       # 调用 predict 生成预测结果
       return  self.predict(output) # (batch_size,seq_len,dec_voc_size)
    
    def encode(self,enc_input):
       enc_embedding = self.pos_encoding(self.enc_embedding(enc_input))
       return self.transformer.encoder(enc_embedding) # (batch_size,seq_len,emb_size)
     
    def decode(self,dec_input,memory,dec_mask):
        dec_embedding = self.pos_encoding(self.dec_embedding(dec_input)) # (batch_size,seq_len,emb_size)  
        return self.transformer.decoder(dec_embedding,memory,dec_mask) # (batch_size,seq_len,emb_size)
      
if __name__ == '__main__':
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>

    # 1. 构建词典

    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    print(chs)
    enc_tokens, dec_tokens = [],[]
    for i in range(1,len(chs)):
      enc = chs[:i]
      dec = ['<s>'] + chs[i:] + ['</s>']
      enc_tokens.append(enc)
      dec_tokens.append(dec)

    print(enc_tokens)
    print(dec_tokens)


      
        