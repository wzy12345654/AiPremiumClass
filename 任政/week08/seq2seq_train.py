import pickle
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from chinese_process import get_proc, read_data
from EncoderDecoder_concat import Seq2Seq
from tqdm import tqdm
# 导入tensorboard进行数据跟踪
from torch.utils.tensorboard import SummaryWriter

# 训练数据
if __name__ == '__main__':

    # 添加设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建Summary对象
    writer = SummaryWriter()
    # 读取词典
    with open('cvocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)
    # 读取编码数据
    with open('cencoder.json', 'r', encoding='utf-8') as f:
        enc_data = json.load(f)
    # 读取解码数据
    with open('cdecoder.json', 'r', encoding='utf-8') as f:
        dec_data = json.load(f)
    # 数据整体json数据集（json）整体压缩成list
    dataset = list(zip(enc_data, dec_data))
    # 转换 DataLoader
    dl = DataLoader(
        dataset,
        batch_size = 256 ,
        shuffle = True,
        collate_fn = get_proc(enc_vocab, dec_vocab)
    )

    # 模型构建
    model = Seq2Seq(
        # 解码数据的长度
        enc_emb_size = len(enc_vocab),
        # 编码数据的长度
        dec_emb_size = len(dec_vocab),
        # 嵌入层维度
        emb_dim = 256,
        # 隐藏层维度
        hidden_size = 256,
        # 失活率
        dropout = 0.5
    )

    # 定义优化器以及损失函数
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # 损失函数 ignore_index 第0个位置不参与计算也就是标签<PAD>不参与计算
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    # 模型训练
    for epoch in range(30):
        # 进度条
        pbar = tqdm(dl)

        model.train()
        for enc_input , dec_input , targets in pbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播
            # 因为该model返回的是元祖  所以我们忽略隐藏状态，所以用_ 当做占位符
            encoder_output = model.encoder(enc_input)
            logins , _ = model(enc_input , dec_input)

            # 计算损失  logins 和 targets 展平
            loss = criterion(logins.view(-1, logins.size(-1)), targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            pbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            # 记录损失
            writer.add_scalar('Loss/train', loss.item(), epoch)

        # 保存模型
        # 在你的 Seq2Seq 模型中，model.state_dict() 会返回包含以下内容的字典：
        # 编码器部分：
        # embedding 层的权重
        # GRU 层的权重和偏置
        # 解码器部分：
        # embedding 层的权重
        # GRU 层的权重和偏置
        # 全连接层的权重和偏置
        torch.save(model.state_dict(), f'seq2seq_model.pth')












