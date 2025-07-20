import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    """
    
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        参数:
            d_model: 词嵌入维度
            dropout:  dropout概率
            max_len:  最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)  # shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 分母项
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos
        
        # 添加batch维度并注册为不参与训练的缓冲区
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入词嵌入序列 (shape: [batch_size, seq_len, d_model])
        返回:
            加入位置编码后的序列
        """
        x = x + self.pe[:, :x.size(1)]  # 截取与输入序列等长的位置编码
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    """
    基于PyTorch的序列到序列Transformer模型
    包含编码器-解码器结构，支持自定义词表大小和模型参数
    """
    def __init__(self, 
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 enc_vocab_size: int = 1000,
                 dec_vocab_size: int = 1000) -> None:
        """
        参数:
            d_model:          模型基础维度（词嵌入维度）
            nhead:            多头注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward:   前馈网络中间层维度
            dropout:          dropout概率
            enc_vocab_size:   编码器词表大小
            dec_vocab_size:   解码器词表大小
        """
        super().__init__()
        # 核心组件初始化
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 启用batch优先的输入格式（[batch, seq, feature]）
        )
        
        # 词嵌入层
        self.enc_embedding = nn.Embedding(enc_vocab_size, d_model)
        self.dec_embedding = nn.Embedding(dec_vocab_size, d_model)
        
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 输出预测层
        self.generator = nn.Linear(d_model, dec_vocab_size)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_pad_mask: Optional[torch.Tensor] = None,
                tgt_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        模型前向传播
        
        参数:
            src:          编码器输入序列 (shape: [batch_size, src_len])
            tgt:          解码器输入序列 (shape: [batch_size, tgt_len])
            tgt_mask:     解码器自注意力掩码（通常为因果掩码） (shape: [tgt_len, tgt_len])
            src_pad_mask: 编码器填充掩码 (shape: [batch_size, src_len])
            tgt_pad_mask: 解码器填充掩码 (shape: [batch_size, tgt_len])
        
        返回:
            预测输出序列 (shape: [batch_size, tgt_len, dec_vocab_size])
        """
        # 词嵌入 + 位置编码
        src_emb = self.pos_encoder(self.enc_embedding(src))
        tgt_emb = self.pos_encoder(self.dec_embedding(tgt))
        
        # 调用Transformer核心计算
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        # 生成词表概率分布
        return self.generator(output)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """编码器前向传播（用于推理阶段）"""
        return self.transformer.encoder(self.pos_encoder(self.enc_embedding(src)))

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """解码器前向传播（用于推理阶段）"""
        return self.transformer.decoder(
            self.pos_encoder(self.dec_embedding(tgt)),
            memory,
            tgt_mask
        )

# ------------------- 以下为模型使用示例 -------------------
class TextDataset(Dataset):
    """简单文本数据集（用于演示）"""
    def __init__(self, text: str) -> None:
        self.chars = list(text)
        self.vocab = sorted(set(self.chars))
        self.char2idx = {c: i+2 for i, c in enumerate(self.vocab)}  # 0:pad, 1:bos, 2:eos
        self.char2idx['<pad>'] = 0
        self.char2idx['<bos>'] = 1
        self.char2idx['<eos>'] = 2
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        
        # 生成训练样本（输入-目标对）
        self.samples = []
        for i in range(1, len(self.chars)):
            src = [self.char2idx[c] for c in self.chars[:i]]
            tgt = [self.char2idx['<bos>']] + [self.char2idx[c] for c in self.chars[i:]] + [self.char2idx['<eos>']]
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.samples[idx][0]), torch.tensor(self.samples[idx][1])

def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """自定义数据整理函数（处理变长序列）"""
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return src_padded, tgt_padded[:, :-1], tgt_padded[:, 1:]  # 输入/目标偏移

if __name__ == '__main__':
    # 超参数设置
    D_MODEL = 256
    NHEAD = 8
    NUM_ENC_LAYERS = 3
    NUM_DEC_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    BATCH_SIZE = 4
    LR = 1e-4

    # 初始化模型
    model = Seq2SeqTransformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        enc_vocab_size=100,  # 实际应根据词表大小调整
        dec_vocab_size=100
    )

    # 初始化数据集和数据加载器
    corpus = "人生得意须尽欢，莫使金樽空对月"
    dataset = TextDataset(corpus)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略pad的损失
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 简单训练循环（演示用）
    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for src, tgt_in, tgt_out in dataloader:
            # 生成因果掩码（防止解码器看到未来信息）
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))
            
            # 前向传播
            output = model(src, tgt_in, tgt_mask)
            
            # 计算损失（展平维度适应交叉熵输入要求）
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
