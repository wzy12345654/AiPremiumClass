from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from torch import nn
from AI.SelfTransformer.TransformerModule import TransformerModel

class DummyTranslationDataset(Dataset):
    """
    虚拟双语数据集
    """

    def __init__(self, num_samples=1000, max_len=20):
        self.src_data = torch.randint(1, 100, (num_samples, max_len))
        self.tgt_data = self.src_data  # 简单复制任务

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # 确保src和tgt的序列长度差为1
        src_seq = self.src_data[idx]
        tgt_seq = self.tgt_data[idx]
        return {
            'src': src_seq,                           # (src_len,)
            'tgt_input': tgt_seq[:-1],                # (tgt_len-1,)
            'tgt_output': tgt_seq[1:]                 # (tgt_len-1,)
        }


def generate_mask(src, tgt, pad_idx=0, device='cpu'):
    """生成正确的掩码格式"""
    # 源序列padding掩码 (batch_size, src_seq_len)
    src_key_padding_mask = (src == pad_idx).to(device)

    # 目标序列padding掩码 (batch_size, tgt_seq_len)
    tgt_key_padding_mask = (tgt == pad_idx).to(device)

    # 解码器look-ahead掩码 (tgt_seq_len, tgt_seq_len)
    look_ahead_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        tgt.size(1)
    ).to(device)

    return None, look_ahead_mask, src_key_padding_mask, tgt_key_padding_mask


# 参数配置
CONFIG = {
    'src_vocab_size': 100,
    'tgt_vocab_size': 100,
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,
    'batch_size': 32,
    'lr': 0.0005,
    'num_epochs': 10,
    'device': "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
}

# 数据管道
dataset = DummyTranslationDataset()
train_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

# 模型初始化
model = TransformerModel(
    src_vocab_size=CONFIG['src_vocab_size'],
    tgt_vocab_size=CONFIG['tgt_vocab_size'],
    d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'],
    num_encoder_layers=CONFIG['num_encoder_layers'],
    num_decoder_layers=CONFIG['num_decoder_layers'],
    dim_feedforward=CONFIG['dim_feedforward'],
    dropout=CONFIG['dropout']
).to(CONFIG['device'])

# 优化器与损失函数
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding计算
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

sample = next(iter(train_loader))
print(f"Source shape: {sample['src'].shape}")       # 应为 [32, 20]
print(f"Target input shape: {sample['tgt_input'].shape}") # 应为 [32, 19]
print(f"Target output shape: {sample['tgt_output'].shape}") # 应为 [32, 19]

# 训练循环
for epoch in range(CONFIG['num_epochs']):
    model.train()
    total_loss = 0

    for batch in train_loader:
        src = batch['src'].to(CONFIG['device'])
        tgt_input = batch['tgt_input'].to(CONFIG['device'])
        tgt_output = batch['tgt_output'].to(CONFIG['device'])

        # 生成掩码
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = generate_mask(
            src, tgt_input, device=CONFIG['device']
        )

        # 前向传播
        output = model(
            src=src,  # 直接传入 (batch, seq)
            tgt=tgt_input,
            src_mask=None,  # 通常不需要源序列掩码
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 计算损失
        loss = criterion(
            output.reshape(-1, output.size(-1)),  # (batch*seq_len, vocab_size)
            tgt_output.reshape(-1)  # (batch*seq_len)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG
}, 'transformer_model.pth')
