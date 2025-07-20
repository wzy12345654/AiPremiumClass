import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from couplet_process import read_couplet_data, get_proc, Vocabulary
from EncoderDecoderAttenModel import Seq2Seq

if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化TensorBoard writer
    writer = SummaryWriter('week08/runs/couplet_training')
    
    # 加载对联数据
    enc_data, dec_data = read_couplet_data('week08/data/fixed_couplets_in.txt', 'week08/data/fixed_couplets_out.txt')
    
    # 构建词表
    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)
    
    # 保存词表
    with open('couplet_vocab.bin', 'wb') as f:
        pickle.dump((enc_vocab, dec_vocab), f)
    
    # 创建数据加载器
    dataset = list(zip(enc_data, dec_data))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=get_proc(enc_vocab, dec_vocab))
    
    # 构建模型
    model = Seq2Seq(
        enc_emb_size=len(enc_vocab),
        dec_emb_size=len(dec_vocab),
        emb_dim=256,
        hidden_size=512,
        dropout=0.5
    )
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=enc_vocab['PAD'])
    
    # 训练循环
    num_epochs = 30
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for enc_input, dec_input, targets in pbar:
            # 数据移至设备
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)
            
            # 前向传播
            logits, _ = model(enc_input, dec_input)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录训练信息到TensorBoard
            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            # 更新进度条和全局步数
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            global_step += 1
        
        # 计算并记录epoch平均损失
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'couplet_model_epoch_{epoch+1}.pt')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'couplet_model_final.pt')
    
    # 关闭TensorBoard writer
    writer.close()