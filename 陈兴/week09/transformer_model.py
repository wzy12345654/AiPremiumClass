import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import os
import numpy as np

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
        pos_embdding = pos_embdding.unsqueeze(0)  # 修改这里，使用0而不是-2
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)  # token长度
        # 确保位置编码的维度与输入张量匹配
        return self.dropout(self.pos_embedding[:, :token_len, :] + token_embdding)

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

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        
    def __len__(self):
        return len(self.enc_tokens)
    
    def __getitem__(self, idx):
        # 将token转换为索引
        enc_indices = [self.enc_vocab[token] for token in self.enc_tokens[idx]]
        dec_indices = [self.dec_vocab[token] for token in self.dec_tokens[idx]]
        
        return {
            'enc_indices': torch.tensor(enc_indices, dtype=torch.long),
            'dec_indices': torch.tensor(dec_indices, dtype=torch.long)
        }

# 数据处理函数
def collate_fn(batch):
    # 获取编码器和解码器的输入序列
    enc_indices = [item['enc_indices'] for item in batch]
    dec_indices = [item['dec_indices'] for item in batch]
    
    # 填充序列
    enc_padded = nn.utils.rnn.pad_sequence(enc_indices, batch_first=True, padding_value=0)
    dec_padded = nn.utils.rnn.pad_sequence(dec_indices, batch_first=True, padding_value=0)
    
    # 创建解码器的输入和目标
    dec_input = dec_padded[:, :-1]  # 去掉最后一个token
    dec_target = dec_padded[:, 1:]  # 去掉第一个token
    
    # 创建mask
    enc_padding_mask = (enc_padded == 0)
    dec_padding_mask = (dec_input == 0)
    
    # 创建解码器的注意力mask（上三角矩阵）
    seq_len = dec_input.size(1)
    tgt_mask = torch.triu(
        torch.ones(seq_len, seq_len) * float('-inf'),
        diagonal=1
    )
    
    return {
        'enc_input': enc_padded,
        'dec_input': dec_input,
        'dec_target': dec_target,
        'tgt_mask': tgt_mask,
        'enc_padding_mask': enc_padding_mask,
        'dec_padding_mask': dec_padding_mask
    }

# 训练函数
def train_model(model, dataloader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # 将数据移动到设备
            enc_input = batch['enc_input'].to(device)
            dec_input = batch['dec_input'].to(device)
            dec_target = batch['dec_target'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            enc_padding_mask = batch['enc_padding_mask'].to(device)
            dec_padding_mask = batch['dec_padding_mask'].to(device)
            
            # 前向传播
            outputs = model(enc_input, dec_input, tgt_mask, enc_padding_mask, dec_padding_mask)
            
            # 计算损失
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), dec_target.reshape(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印每个epoch的平均损失
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# 保存模型函数
def save_model(model, optimizer, vocab, save_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'enc_vocab': vocab['enc_vocab'],
        'dec_vocab': vocab['dec_vocab']
    }
    torch.save(checkpoint, save_path)
    print(f'模型已保存到 {save_path}')

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
    
    # 构建encoder和decoder的词典
    special_tokens = ['<pad>', '<s>', '</s>']
    all_tokens = special_tokens + list(set(chs))
    
    enc_vocab = {token: idx for idx, token in enumerate(all_tokens)}
    dec_vocab = {token: idx for idx, token in enumerate(all_tokens)}
    
    # 创建数据集和数据加载器
    dataset = TextDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    d_model = 128  # 嵌入维度
    nhead = 8  # 注意力头数
    num_enc_layers = 3  # 编码器层数
    num_dec_layers = 3  # 解码器层数
    dim_forward = 512  # 前馈网络维度
    dropout = 0.1  # Dropout率
    enc_voc_size = len(enc_vocab)  # 编码器词汇表大小
    dec_voc_size = len(dec_vocab)  # 解码器词汇表大小
    
    # 创建模型
    model = Seq2SeqTransformer(
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        dim_forward=dim_forward,
        dropout=dropout,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=enc_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练模型
    epochs = 100
    train_model(model, dataloader, optimizer, criterion, device, epochs)
    
    # 保存模型
    save_dir = '/Users/chenxing/AI/AiPremiumClass/陈兴/week09/models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'transformer_model.pth')
    
    vocab = {
        'enc_vocab': enc_vocab,
        'dec_vocab': dec_vocab
    }
    
    save_model(model, optimizer, vocab, save_path)

