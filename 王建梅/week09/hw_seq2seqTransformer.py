import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        # math.sqrt(self.emb_size)是常规缩放操作，有助于在训练过程中稳定梯度
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self,
                emb_size: int,
                dropout: float,
                maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 奇数行、偶数行词向量值分别对应正弦和余弦
        # 10000** (2i/d_model)
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 奇数⾏、偶数⾏词向量值分别对应正弦和余弦
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 为和TokenEmbedding对⻬⽽增加的维度 [maxlen,1,emb_size]
        pos_embedding = pos_embedding.unsqueeze(-2)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 把positional embedding注册为⼀个缓冲,不需要作为模型参数进⾏训练
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        # token embedding和positional embedding相加后再经过dropout过滤，相加用到了广播机制,相加后的形状为[batch_size,seq_len,emb_size]
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 emb_size: int, 
                 nhead: int, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 dim_feedforward: int, 
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout,
                                          batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, 
                src, 
                tgt,
                tgt_mask, 
                src_padding_mask, 
                tgt_padding_mask):
        # 对源和目标序列进行词嵌入和位置编码
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        # 调用transformer模型进行编码和解码(解码加了掩码)
        outs = self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask)
        return self.generator(outs)
    
    def encode(self, src):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)))
    
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


if __name__ == '__main__':
    emb_size = 8
    seq_len = 6
    # emb = nn.Embedding(num_emb, emb_size, _weight=torch.ones(num_emb, emb_size))
    # pos_enc = PositionalEncoding(emb_size, 0)

    torch.manual_seed(42)  # 设置随机种子以确保结果可复现

    # 测试位置编码，seq_len=5，emb_size=8 
    tokenEmbedding = TokenEmbedding(seq_len, emb_size)
    posEmbedding = PositionalEncoding(emb_size, 0)

    token_inputs = torch.range(0,5, dtype=torch.long)
    #print(token_inputs)
    token_embedding = tokenEmbedding(token_inputs)
    print(token_embedding.shape)
    postion_Embedding = posEmbedding(token_embedding)
    print(postion_Embedding.shape)