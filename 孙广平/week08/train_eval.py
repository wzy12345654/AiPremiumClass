import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from config import config  
from data_loader import CoupletDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq

def train_model():
    # 初始化数据集
    train_dataset = CoupletDataset('./couplet/train/in.txt', './couplet/train/out.txt')
    test_dataset = CoupletDataset('./couplet/test/in.txt', './couplet/test/out.txt', vocab=train_dataset.vocab)
    
    # 拆分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, 
                            batch_size=config['batch_size'], 
                            collate_fn=collate_fn, 
                            shuffle=True)
    val_loader = DataLoader(val_subset, 
                          batch_size=config['batch_size'], 
                          collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, 
                           batch_size=config['batch_size'], 
                           collate_fn=collate_fn)
    
    # 初始化模型（修正设备访问方式）
    encoder = Encoder(len(train_dataset.vocab)).to(config['device'])  
    decoder = Decoder(len(train_dataset.vocab)).to(config['device'])
    model = Seq2Seq(encoder, decoder).to(config['device'])
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    writer = SummaryWriter()
    
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(config['n_epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for src, src_len, trg in progress_bar:
            optimizer.zero_grad()
            output = model(src, src_len, trg)
            
            # 计算损失（忽略<pad>）
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            
            # 反向传播（修正梯度裁剪参数访问）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])  
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, criterion)
        writer.add_scalars('Loss', {'train': epoch_loss/len(train_loader), 'val': val_loss}, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.3f} | Val Loss: {val_loss:.3f}')
    
    # 测试评估
    model.load_state_dict(torch.load('best_model.pth', map_location=config['device']))  
    model = model.to(config['device'])
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f}')

def evaluate(model, loader, criterion):
    """模型评估函数"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, src_len, trg in loader:
            src = src.to(config['device'])
            trg = trg.to(config['device'])
            
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    train_model()