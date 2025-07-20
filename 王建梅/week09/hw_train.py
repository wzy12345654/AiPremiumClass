import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hw_process import get_proc,create_mask
import hw_seq2seqTransformer as est
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# Seq2SeqTransformer model for training



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1 准备数据集
# 1.加载数据集 & 定义模型、损失函数和优化器
with open('data/model/hw_train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('data/model/hw_vocab.bin','rb') as f:
    vocab = pickle.load(f)

# 超参数设置
batch_size = 2
num_epochs = 30
learning_rate = 0.001
num_encoder_layers = 2
num_decoder_layers = 2
emb_size = 128
nhead = 2
dim_feedforward = 256  # 前馈神经网络的隐藏层维度
src_vocab_size = len(vocab)  # 输入词典大小
tgt_vocab_size = len(vocab)  # 输出词典大小

# 2 创建模型
model = est.Seq2SeqTransformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                         emb_size=emb_size, nhead=nhead, 
                                         src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                         dim_feedforward=dim_feedforward)
model.to(device)

# 3 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充索引为0的损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4 训练模型&保存模型
writer = SummaryWriter()
train_loss_cnt = 0
model.train()
# 创建数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size, 
                          shuffle=True,
                          collate_fn=get_proc(vocab,vocab))

for epoch in range(num_epochs):
    model.train()
    tpbar = tqdm(train_loader)
    for src, tgt in tpbar:
        src = src.to(device)  # 将输入数据转移到GPU上
        tgt = tgt.to(device)  # 将输入数据转移到GPU上

        tgt_input = tgt[:, :-1]  # 解码器输入不包含最后一个token
        tgt_output = tgt[:, 1:]  # 解码器输出不包含第一个token
        # 生成掩码矩阵, 用于屏蔽填充位置的损失
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)

        # 前向传播
        outputs = model(src, tgt_input,tgt_mask,src_padding_mask, tgt_padding_mask)  
        # 计算损失, 解码器输入不包含第一个token, 解码器输出不包含最后一个token
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
        train_loss_cnt += 1

    # 保存模型，因为训练时间太长，每轮训练后保存模型
    torch.save(model.state_dict(), 'data/model/hw_seq2seqTransformer_model.bin')