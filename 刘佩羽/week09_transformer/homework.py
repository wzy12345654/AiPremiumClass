import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置环境变量以允许程序继续执行
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(
            -torch.arange(0, emb_size, 2)*math.log(10000)/emb_size
        )
        # 位置编码索引 shape = (maxlen, 1)
        pos = torch.arange(0, maxlen).reshape(-1, 1)
        # 编码矩阵(maxlen, emb_size)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:,0::2] = torch.sin(pos*den)
        pos_embedding[:,1::2] = torch.cos(pos*den)
        # 添加一个和batch对应的维度
        pos_embedding = pos_embedding.unsqueeze(0)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding: [batch_size, seq_len, embedding_dim]
        token_len = token_embedding.size(1)
        # 添加位置编码
        add_emb = token_embedding + self.pos_embedding[:, :token_len, :]
        return self.dropout(add_emb)


class Seq2SeqTransformer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            enc_voc_size,
            dec_voc_size
    ):
        super().__init__()

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # encoder input embedding
        self.enc_emb = nn.Embedding(
            num_embeddings=enc_voc_size,
            embedding_dim=d_model
        )
        # decoder input embedding
        self.dec_emb = nn.Embedding(
            num_embeddings=dec_voc_size,
            embedding_dim=d_model
        )
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)

        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self, enc_input, dec_input,
            tgt_mask=None, src_key_padding_mask=None,
            tgt_key_padding_mask=None
    ):
        # multi head attention 之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_input))
        dec_emb = self.pos_encoding(self.dec_emb(dec_input))

        # 调用transformer计算
        outs = self.transformer(
            src=enc_emb,
            tgt=dec_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.predict(outs)

    # 推理阶段使用的方法
    def encode(self, enc_input, src_key_padding_mask=None):
        enc_emb = self.pos_encoding(self.enc_emb(enc_input))
        return self.transformer.encoder(enc_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, dec_input, memory, tgt_mask=None, tgt_key_padding_mask=None):
        dec_emb = self.pos_encoding(self.dec_emb(dec_input))
        return self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol, end_symbol):
        """
        贪婪解码方法用于推理
        """
        device = src.device
        batch_size = src.shape[0]
        
        # 编码器处理输入序列
        memory = self.encode(src, src_mask)
        
        # 初始化解码序列
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            # 准备解码器的掩码
            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)
            
            # 解码当前序列
            out = self.decode(ys, memory, tgt_mask)
            out = self.predict(out)
            
            # 获取下一个预测token
            _, next_word = torch.max(out[:, -1], dim=1)
            next_word = next_word.unsqueeze(1)
            
            # 将预测添加到解码序列
            ys = torch.cat([ys, next_word], dim=1)
            
            # 如果所有序列都预测了结束符，提前结束
            if (next_word == end_symbol).all():
                break
                
        return ys
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """生成用于解码器的掩码（上三角为-inf）"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


class TransformerDataset(Dataset):
    def __init__(self, corpus, seq_length=None):
        self.corpus = corpus
        self.seq_length = seq_length if seq_length else len(corpus)
        
        # 创建词汇表
        self.chars = ['<s>'] + sorted(list(set(corpus))) + ['</s>']
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # 特殊标记
        self.start_symbol = self.char_to_idx['<s>']
        self.end_symbol = self.char_to_idx['</s>']
        
        # 准备训练数据
        self.enc_tokens, self.dec_tokens_input, self.dec_tokens_target = self._prepare_data()
    
    def _prepare_data(self):
        enc_tokens = []
        dec_tokens_input = []  # 以<s>开头
        dec_tokens_target = []  # 以</s>结尾
        
        # 生成所有可能的训练样本
        for i in range(1, len(self.corpus)):
            # 编码器输入: 前i个字符
            enc_input = [self.char_to_idx[ch] for ch in self.corpus[:i]]
            
            # 解码器输入: <s> + 后面的字符
            dec_input = [self.start_symbol] + [self.char_to_idx[ch] for ch in self.corpus[i:]]
            
            # 解码器目标: 后面的字符 + </s>
            dec_target = [self.char_to_idx[ch] for ch in self.corpus[i:]] + [self.end_symbol]
            
            enc_tokens.append(enc_input)
            dec_tokens_input.append(dec_input)
            dec_tokens_target.append(dec_target)
        
        return enc_tokens, dec_tokens_input, dec_tokens_target
    
    def __len__(self):
        return len(self.enc_tokens)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.enc_tokens[idx], dtype=torch.long),
            torch.tensor(self.dec_tokens_input[idx], dtype=torch.long),
            torch.tensor(self.dec_tokens_target[idx], dtype=torch.long)
        )


def collate_fn(batch):
    """自定义数据批次处理函数"""
    # 分离编码器输入、解码器输入和目标
    enc_inputs, dec_inputs, dec_targets = zip(*batch)
    
    # 填充序列
    enc_inputs_padded = nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True, padding_value=0)
    dec_inputs_padded = nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)
    dec_targets_padded = nn.utils.rnn.pad_sequence(dec_targets, batch_first=True, padding_value=0)
    
    # 创建填充掩码（0表示填充位置）
    enc_padding_mask = (enc_inputs_padded == 0)
    dec_padding_mask = (dec_inputs_padded == 0)
    
    return enc_inputs_padded, dec_inputs_padded, dec_targets_padded, enc_padding_mask, dec_padding_mask


def create_tgt_mask(size):
    """创建目标序列的掩码：不允许解码器看到未来的标记"""
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = 0
    
    for enc_inputs, dec_inputs, dec_targets, enc_padding_mask, dec_padding_mask in dataloader:
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        dec_targets = dec_targets.to(device)
        enc_padding_mask = enc_padding_mask.to(device)
        dec_padding_mask = dec_padding_mask.to(device)
        
        # 创建目标掩码
        tgt_mask = create_tgt_mask(dec_inputs.size(1)).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(
            enc_input=enc_inputs,
            dec_input=dec_inputs,
            tgt_mask=tgt_mask,
            src_key_padding_mask=enc_padding_mask,
            tgt_key_padding_mask=dec_padding_mask
        )
        
        # 计算损失（忽略填充）
        output = output.reshape(-1, output.shape[-1])
        dec_targets = dec_targets.reshape(-1)
        loss = criterion(output, dec_targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
    
    return losses / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = 0
    
    with torch.no_grad():
        for enc_inputs, dec_inputs, dec_targets, enc_padding_mask, dec_padding_mask in dataloader:
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_targets = dec_targets.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)
            
            # 创建目标掩码
            tgt_mask = create_tgt_mask(dec_inputs.size(1)).to(device)
            
            # 前向传播
            output = model(
                enc_input=enc_inputs,
                dec_input=dec_inputs,
                tgt_mask=tgt_mask,
                src_key_padding_mask=enc_padding_mask,
                tgt_key_padding_mask=dec_padding_mask
            )
            
            # 计算损失
            output = output.reshape(-1, output.shape[-1])
            dec_targets = dec_targets.reshape(-1)
            loss = criterion(output, dec_targets)
            
            losses += loss.item()
    
    return losses / len(dataloader)


def generate_text(model, dataset, seed_text, max_length=10, device='cpu'):
    model.eval()
    
    # 将种子文本转换为模型的输入形式
    input_indices = [dataset.char_to_idx[ch] for ch in seed_text]
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
    
    # 初始化解码序列
    decoder_input = torch.tensor([[dataset.start_symbol]], dtype=torch.long).to(device)
    
    # 编码器处理输入
    encoder_outputs = model.encode(input_tensor)
    
    # 逐个字符生成输出
    output_text = []
    for i in range(max_length):
        # 创建掩码
        tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        
        # 解码
        decoder_output = model.decode(decoder_input, encoder_outputs, tgt_mask)
        decoder_output = model.predict(decoder_output)
        
        # 获取最后一个位置的预测
        prediction = decoder_output[:, -1, :]
        _, next_char_idx = torch.max(prediction, dim=1)
        
        # 如果预测到结束符，停止生成
        if next_char_idx.item() == dataset.end_symbol:
            break
            
        # 将预测的字符添加到输出
        output_text.append(dataset.idx_to_char[next_char_idx.item()])
        
        # 将预测添加到解码输入，用于下一步预测
        decoder_input = torch.cat([decoder_input, next_char_idx.unsqueeze(0)], dim=1)
    
    return ''.join(output_text)


def main():
    # 语料
    corpus = "人生得意须尽欢，莫使金樽空对月"
    
    # 超参数
    d_model = 32  # 较小的模型维度，适合小数据集
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    batch_size = 2
    epochs = 200
    learning_rate = 0.001
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset = TransformerDataset(corpus)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Characters mapping: {dataset.char_to_idx}")
    
    # 创建模型
    model = Seq2SeqTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        enc_voc_size=dataset.vocab_size,
        dec_voc_size=dataset.vocab_size
    ).to(device)
    
    # 打印模型结构
    print(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记(0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    train_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 生成文本示例
    seed_text = "人生得意"
    generated_text = generate_text(model, dataset, seed_text, max_length=15, device=device)
    print(f"Seed: {seed_text}")
    print(f"Generated: {generated_text}")
    
    # 保存模型
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model saved to 'transformer_model.pth'")


if __name__ == '__main__':
    main()
