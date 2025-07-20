import pickle
import torch
import json
from torch.utils.data import DataLoader
from process import get_proc, Vocabulary
from EncoderDecoderAttenModel import Seq2Seq # type: ignore
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def load_model():
    """
    加载模型，准备训练数据，执行模型训练，并保存训练好的模型状态字典。

    此函数会完成以下操作：
    1. 检测可用的计算设备（CUDA、MPS 或 CPU）。
    2. 初始化 TensorBoard 的 SummaryWriter 用于记录训练损失。
    3. 加载词汇表和训练数据。
    4. 构建 Seq2Seq 模型并将其移动到指定设备。
    5. 初始化优化器和损失函数。
    6. 执行多轮训练，并在训练过程中更新模型参数。
    7. 使用 TensorBoard 记录训练损失。
    8. 保存训练好的模型状态字典。
    """
    try:
        # 检查 MPS (Apple Silicon GPU) 是否可用
        mps_available = torch.mps.is_available()
    except AttributeError:
        # 若没有 mps.is_available 方法，说明 MPS 不可用
        mps_available = False
    
    # 创建 SummaryWriter 对象，用于将训练信息写入 TensorBoard 日志文件
    writer = SummaryWriter()
    # 初始化训练损失计数器，用于记录训练步数
    train_loss_cnt = 0
    # 根据设备可用性选择计算设备，优先使用 CUDA，其次是 MPS，最后是 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if mps_available else 'cpu'))

    # 加载训练数据
    # 词汇表文件路径
    vocab_file = 'week8_codes/couplet/vocabs'
    # 从文件中加载词汇表
    vocab = Vocabulary.from_file(vocab_file)

    # 加载编码器数据
    with open('week8_codes/encoder.json') as f:
        enc_data = json.load(f)
    # 加载解码器数据
    with open('week8_codes/decoder.json') as f:
        dec_data = json.load(f)

    # 将编码器数据和解码器数据组合成数据集
    ds = list(zip(enc_data, dec_data))
    # 创建数据加载器，设置批量大小为 256，数据打乱顺序，使用自定义的 collate 函数
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=get_proc(vocab.vocab))

    # 构建训练模型
    # 初始化 Seq2Seq 模型
    model = Seq2Seq(
        enc_emb_size=len(vocab.vocab),  # 编码器嵌入层的输入维度，即词汇表大小
        dec_emb_size=len(vocab.vocab),  # 解码器嵌入层的输入维度，即词汇表大小
        emb_dim=200,  # 嵌入层的维度
        hidden_size=250,  # 隐藏层的维度
        dropout=0.5,  # Dropout 概率，防止过拟合
        # state_type="add"  # 编码器隐藏状态的处理方式，可按需取消注释
    )
    # 将模型移动到指定的计算设备上
    model.to(device)

    # 优化器、损失
    # 使用 Adam 优化器，设置学习率和权重衰减
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练
    # 训练 20 个轮次
    for epoch in range(20):
        # 将模型设置为训练模式
        model.train()
        # 使用 tqdm 显示训练进度条
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            # 将编码器输入数据移动到指定设备
            enc_input = enc_input.to(device)
            # 将解码器输入数据移动到指定设备
            dec_input = dec_input.to(device)
            # 将目标数据移动到指定设备
            targets = targets.to(device)

            # 前向传播 
            # 调用模型进行前向计算，得到预测的 logits 和其他信息
            logits, _ = model(enc_input, dec_input)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits 形状: [batch_size, seq_len, vocab_size]
            # targets 形状: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # 反向传播
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 计算损失的梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            # 更新进度条信息，显示当前轮次和损失值
            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            # 使用 TensorBoard 记录训练损失
            writer.add_scalar('week8_cofes/Loss/train', loss.item(), train_loss_cnt)
            # 训练步数加 1
            train_loss_cnt += 1

    # 保存训练好的模型状态字典到文件
    torch.save(model.state_dict(), 'seq2seq_state.bin')
    
    
if __name__ == '__main__':
    load_model()