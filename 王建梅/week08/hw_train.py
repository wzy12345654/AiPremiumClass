import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from hw_EncoderDecoderAttention import Encoder,Decoder,Attention,Seq2Seq
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from hw_process import get_proc

# 1.加载数据集 & 定义模型、损失函数和优化器
with open('data/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
# with open('data/test_dataset.pkl', 'rb') as f:
#     test_dataset = pickle.load(f)
with open('data/vocab.bin','rb') as f:
    vocab = pickle.load(f)
   

# 定义超参数
input_dim = len(vocab)  # 输入词典大小
output_dim = len(vocab)  # 输出词典大小
embedding_dim = 128  # 词嵌入维度
hidden_dim = 256  # 隐藏层维度
dropout = 0.5  # dropout概率
batch_size = 512  # 批次大小
num_epochs = 10  # 训练轮数
learning_rate = 0.001  # 学习率

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
train_loss_cnt = 0
# 创建模型、损失函数和优化器
model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim,dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 创建数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size, 
                          shuffle=True,
                          collate_fn=get_proc(vocab,vocab))

# 2. 训练模型
for epoch in range(num_epochs):
    model.train()
    tpbar = tqdm(train_loader)
    for enc_input, dec_input, targets in tpbar:
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        targets = targets.to(device)

        # 前向传播
        output, _ = model(enc_input, dec_input)  # 忽略decoder的隐藏状态
        
        # 计算损失
        # CrossEntropyLoss需要将logits和targets展平
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
        train_loss_cnt += 1

    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'data/model/seq2seq_model.bin')  # 一轮训练用了1小时，没训练完一轮，保存一次模型参数，防止训练中断

# 3. 保存模型
# torch.save(model.state_dict(), 'data/model/seq2seq_model.bin')
# 4. 评估模型
# model.eval()
# total_loss = 0
# with torch.no_grad():
#     for epoch in range(num_epochs):
#         tpbar = tqdm(test_loader)
#         for enc_input, dec_input, targets in tpbar:
#             enc_input = enc_input.to(device)
#             dec_input = dec_input.to(device)
#             targets = targets.to(device)
#             # 前向传播
#             output, _ = model(enc_input, dec_input)  # 忽略decoder的隐藏状态
#             # 计算损失
#             loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
#             total_loss += loss.item()
#             tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
#     avg_loss = total_loss / len(test_loader)
#     print(f'Test Loss: {avg_loss:.4f}')

            