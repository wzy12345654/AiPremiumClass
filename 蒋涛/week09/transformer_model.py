import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# 位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        """
        初始化位置编码模块。

        参数:
        emb_size (int): 词嵌入的维度，位置编码的维度需与词嵌入维度一致。
        dropout (float): Dropout 层的丢弃概率，用于防止过拟合。
        maxlen (int, optional): 最大序列长度，默认为 5000。
        """
        super().__init__()
        # 计算行缩放指数值，用于后续正弦和余弦函数的计算。
        # 公式为 exp(-2i * log(10000) / emb_size)，其中 i 为偶数索引
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 生成位置编码的索引，形状为 (maxlen, 1)，表示每个位置的索引
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 初始化位置编码矩阵，形状为 (maxlen, emb_size)，初始值全为 0
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 为位置编码矩阵的偶数索引位置填充正弦函数值
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        # 为位置编码矩阵的奇数索引位置填充余弦函数值
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 为位置编码矩阵添加一个批次维度，形状变为 (1, maxlen, emb_size)
        # 以便后续与批次数据进行广播操作
        pos_embedding = pos_embedding.unsqueeze(0)
        # 定义 Dropout 层，用于在训练过程中随机丢弃部分神经元
        self.dropout = nn.Dropout(dropout)
        # 将位置编码矩阵注册为缓冲区，使其作为模型状态的一部分，但不参与反向传播更新
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        """
        前向传播方法，将词嵌入与位置编码相加，并应用 Dropout 操作。

        参数:
        token_embedding (torch.Tensor): 词嵌入张量，形状为 [batch_size, seq_len, embedding_dim]。

        返回:
        torch.Tensor: 经过位置编码和 Dropout 处理后的张量，形状与输入相同。
        """
        # token_embedding: [batch_size, seq_len, embedding_dim]
        # 从位置编码矩阵中截取与输入词嵌入序列长度相同的部分
        # 然后将其与词嵌入张量相加，为词嵌入添加位置信息
        # self.pos_embedding[:, :token_embedding.size(1), :] 的形状为 [1, seq_len, embedding_dim]
        # 利用广播机制与 token_embedding 相加
        pos_embedded = token_embedding + self.pos_embedding[:, :token_embedding.size(1), :]
        # 对添加了位置编码的张量应用 Dropout 操作，防止过拟合
        return self.dropout(pos_embedded)

class Seq2SeqTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        """
        初始化 Seq2SeqTransformer 模型。

        参数:
        d_model (int): 模型的特征维度，即词嵌入的维度。
        nhead (int): 多头注意力机制中的头数。
        num_enc_layers (int): 编码器的层数。
        num_dec_layers (int): 解码器的层数。
        dim_forward (int): 前馈神经网络的隐藏层维度。
        dropout (float): Dropout 层的丢弃概率，用于防止过拟合。
        enc_voc_size (int): 编码器输入词汇表的大小。
        dec_voc_size (int): 解码器输出词汇表的大小。
        """
        super().__init__()
        # 初始化 Transformer 模型
        # batch_first=True 表示输入和输出的张量维度中，批次维度在第一维
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # 编码器输入的词嵌入层，将编码器输入的 token 索引转换为词向量
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        # 解码器输入的词嵌入层，将解码器输入的 token 索引转换为词向量
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # 预测层，将 Transformer 输出的特征向量映射到解码器词汇表的大小，用于预测 token
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # 位置编码层，为输入的词嵌入添加位置信息
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        """
        前向传播方法，定义 Seq2SeqTransformer 模型的计算流程。

        参数:
        enc_inp (torch.Tensor): 编码器的输入张量，形状为 [batch_size, enc_seq_len]。
        dec_inp (torch.Tensor): 解码器的输入张量，形状为 [batch_size, dec_seq_len]。
        tgt_mask (torch.Tensor): 解码器的后续掩码，用于防止模型看到未来的信息，形状为 [dec_seq_len, dec_seq_len]。
        enc_pad_mask (torch.Tensor): 编码器输入的填充掩码，用于标记填充位置，形状为 [batch_size, enc_seq_len]。
        dec_pad_mask (torch.Tensor): 解码器输入的填充掩码，用于标记填充位置，形状为 [batch_size, dec_seq_len]。

        返回:
        torch.Tensor: 模型的预测输出，形状为 [batch_size, dec_seq_len, dec_voc_size]。
        """
        # 对编码器输入的 token 索引进行词嵌入，再添加位置编码信息
        # enc_emb 形状为 [batch_size, enc_seq_len, d_model]
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        # 对解码器输入的 token 索引进行词嵌入，再添加位置编码信息
        # dec_emb 形状为 [batch_size, dec_seq_len, d_model]
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用 PyTorch 的 Transformer 模块进行计算
        # src 为编码器的输入，tgt 为解码器的输入
        # tgt_mask 防止解码器看到未来信息，src_key_padding_mask 和 tgt_key_padding_mask 用于忽略填充位置
        # outs 形状为 [batch_size, dec_seq_len, d_model]
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 将 Transformer 的输出通过全连接层进行映射，得到每个位置的 token 预测概率
        # 返回的张量形状为 [batch_size, dec_seq_len, dec_voc_size]
        return self.predict(outs)

    # 推理环节使用方法
    def encode(self, enc_inp):
        """
        对编码器输入进行编码处理，将输入的 token 索引转换为编码后的特征表示。

        参数:
        enc_inp (torch.Tensor): 编码器的输入张量，形状为 [batch_size, enc_seq_len]，
                                包含一批样本的 token 索引。

        返回:
        torch.Tensor: 编码器输出的特征张量，形状为 [batch_size, enc_seq_len, d_model]，
                      表示输入序列经过编码器处理后的特征表示。
        """
        # 对编码器输入的 token 索引进行词嵌入操作，将其转换为词向量
        # 再通过位置编码层为词向量添加位置信息
        # enc_emb 形状为 [batch_size, enc_seq_len, d_model]
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        # 将添加了位置编码的词向量输入到 Transformer 编码器中进行编码
        # 返回编码器输出的特征张量
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        """
        对解码器输入进行解码处理，结合编码器的输出得到解码后的特征表示。

        参数:
        dec_inp (torch.Tensor): 解码器的输入张量，形状为 [batch_size, dec_seq_len]，包含一批样本的 token 索引。
        memory (torch.Tensor): 编码器的输出张量，形状为 [batch_size, enc_seq_len, d_model]，作为解码器的记忆信息。
        dec_mask (torch.Tensor): 解码器的掩码张量，用于控制注意力机制，形状通常为 [dec_seq_len, dec_seq_len]。

        返回:
        torch.Tensor: 解码器输出的特征张量，形状为 [batch_size, dec_seq_len, d_model]，表示解码后的特征表示。
        """
        # 对解码器输入的 token 索引进行词嵌入操作，将其转换为词向量
        # 再通过位置编码层为词向量添加位置信息
        # dec_emb 形状为 [batch_size, dec_seq_len, d_model]
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 将添加了位置编码的解码器输入词向量、编码器的输出（memory）以及解码器掩码输入到 Transformer 解码器中进行解码
        # 返回解码器输出的特征张量
        return self.transformer.decoder(dec_emb, memory, dec_mask)
 
if __name__ == '__main__':
    
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "风弦未拨心先乱，夜幕已沉梦更闲"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    
    print(enc_tokens)
    print(dec_tokens)



