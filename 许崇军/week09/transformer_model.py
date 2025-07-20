import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.nn.utils.rnn import pad_sequence
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
        pos_embdding = pos_embdding.unsqueeze(0)
        #
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)
    def forward(self, token_embdding):
        token_len = token_embdding.size(1)  # token长度。size(1)是第二个维度
        # (1, token_len, emb_size)
        add_emb = self.pos_embedding[:,:token_len, :] + token_embdding
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

    def forward(self, enc_inp, dec_inp, tgt_mask, src_padding_mask, tgt_padding_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask)
        # 推理
        return self.predict(outs)

    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)

def generate_square_subsequent_mask(sz):
    """生成上三角mask，屏蔽未来token"""
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def create_mask(src, tgt, pad_idx=0):
    """
    src: [B, S]
    tgt: [B, T]
    """
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    # Decoder 自注意力的 mask (causal)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    # Encoder & Decoder 的 padding mask
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    src_padding_mask = src_padding_mask.to(torch.bool)
    tgt_padding_mask = tgt_padding_mask.to(torch.bool)
    return tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == '__main__':

    d_model = 512  # 模型的维度
    nhead = 8  # 多头注意力机制的头数
    num_enc_layers = 3  # 编码器的层数
    num_dec_layers = 3  # 解码器的层数
    dim_forward = 2048  # 前馈神经网络的维度
    dropout = 0.1  # dropout概率
    enc_voc_size = 1000  # 编码器词汇表的大小
    dec_voc_size = 1000  # 解码器词汇表的大小
    # 创建模型实例
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size,
                               dec_voc_size)

    # 随机生成输入数据，假设batch size为2，src sequence length为10，tgt sequence length为15
    src = torch.randint(0, enc_voc_size, (2, 10))  # (batch_size, src_seq_length)
    tgt = torch.randint(0, dec_voc_size, (2, 15))  # (batch_size, tgt_seq_length)

    # 生成tgt_mask, src_padding_mask, tgt_padding_mask
    tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(src.device)
    src_padding_mask = (src == 0)  # 假设0是padding index
    tgt_padding_mask = (tgt == 0)  # 假设0是padding index

    # 前向传播测试
    output = model(src, tgt, tgt_mask, src_padding_mask, tgt_padding_mask)
    print("Output Shape:", output.shape)  # 应该是(batch_size, tgt_seq_length, dec_voc_size)

    # encode测试
    memory = model.encode(src)
    print("Memory Shape:", memory.shape)  # 应该是(batch_size, src_seq_length, d_model)

    # decode测试
    dec_out = model.decode(tgt, memory, tgt_mask)
    print("Decoder Output Shape:", dec_out.shape)  # 应该是(batch_size, tgt_seq_length, d_model)

    # 预测测试
    predictions = model.predict(dec_out)
    print("Predictions Shape:", predictions.shape)  # 应该是(batch_size, tgt_seq_length, dec_voc_size)