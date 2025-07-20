import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # 初始化PositionalEncoding类
        super(PositionalEncoding, self).__init__()
        
        # 创建一个全为0的矩阵，大小为max_len * d_model
        pe = torch.zeros(max_len, d_model)
        # 创建一个从0到max_len的向量，大小为max_len * 1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建一个向量，大小为d_model / 2，每个元素为exp(-log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 将position向量乘以div_term向量，并将结果赋值给pe矩阵的偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        # 将position向量乘以div_term向量，并将结果赋值给pe矩阵的奇数列
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将pe矩阵在第0维上增加一个维度，大小为1 * max_len * d_model
        pe = pe.unsqueeze(0)
        # 将pe矩阵注册为缓冲区，使其在模型保存和加载时被保存和加载
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 将输入x与位置编码相加
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 添加此行
        )
        
        self.fc_out = nn.Linear(d_model, input_dim)
        
    def forward(self, src, tgt):
        # Embedding和位置编码
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        
        # Transformer处理
        transformer_output = self.transformer(src_emb, tgt_emb)
        
        # 输出层
        output = self.fc_out(transformer_output)
        
        return output
    
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类（示例）
class TranslationDataset(Dataset):
    def __init__(self, data):
        # 初始化函数，用于创建类的实例
        self.data = data
        
    # 定义一个方法，用于返回对象的长度
    def __len__(self):
        # 返回对象的data属性的长度
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取索引为idx的数据
        src, tgt = self.data[idx]
        # 将src和tgt转换为torch.tensor类型
        return torch.tensor(src), torch.tensor(tgt)

# 超参数
input_dim = 10000  # 词汇表大小
batch_size = 32
learning_rate = 0.0001
num_epochs = 10

# 初始化模型、优化器和损失函数
model = TransformerModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是padding索引

# 简单示例数据（应替换为实际数据）
def create_sample_data(num_samples=1000):
    # 这里只是一个示例，应该用真实的翻译数据替代
    src_data = [list(range(5)) for _ in range(num_samples)]  # 假设源句子长度为5
    tgt_data = [list(range(5)) for _ in range(num_samples)]  # 假设目标句子长度为5
    return list(zip(src_data, tgt_data))

# 创建数据集和数据加载器
dataset = TranslationDataset(create_sample_data())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建数据加载器
# dataset = TranslationDataset(...)  # 这里需要实际的数据
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练循环
# 训练循环修正版
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # 确保dataloader已定义并初始化
    for src, tgt in dataloader:
        optimizer.zero_grad()
        
        # 前向传播 - 注意调整输入维度以匹配batch_first=True
        output = model(src, tgt[:, :-1])  # 解码器输入要移位
        
        # 计算损失 - 注意调整目标维度以匹配输出形状
        loss = criterion(output.view(-1, input_dim), tgt[:, 1:].contiguous().view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
