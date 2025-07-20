import torch
from transformer_seq2seq_model import Seq2SeqTransformer, PositionalEncoding
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_fix import read_data, get_proc, Vocabulary
import pickle

if __name__ == '__main__':
    
    writer = SummaryWriter()
    train_loss_cnt = 0
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    ### 加载训练数据

    vocab_file = '/mnt/data_1/zfy/self/homework/kagglehub/datasets/jiaminggogogo/chinese-couplets/versions/2/couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)



    with open('encoder.json') as f:
        enc_data = json.load(f)
    with open('decoder.json') as f:
        dec_data = json.load(f)
    ds = list(zip(enc_data,dec_data))
    dl = DataLoader(ds, batch_size=512, shuffle=True,num_workers=4, collate_fn=get_proc(vocab))


    # 构建训练模型
    # 模型构建
    model = Seq2SeqTransformer(
        d_model=64,            # 每个 token 的嵌入维度
        nhead=4,               # 多头注意力中头的数量
        num_enc_layers=2,      # 编码器层数
        num_dec_layers=2,      # 解码器层数
        dim_forward=256,       # FFN 维度
        dropout=0.1,           # dropout 概率
        enc_voc_size=len(vocab),  # 编码器词表大小
        dec_voc_size=len(vocab)   # 解码器词表大小
    )

    model.to(device)
    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    # 训练
    # 初始化全局步骤计数器
    global_step = 0
    num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        tpbar = tqdm(dl, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (enc_input, dec_input, targets) in enumerate(tpbar):
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播
            
            # 构造 tgt_mask（即 Transformer 解码器的自回归 mask）
            tgt_seq_len = dec_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(dec_input.size(1)).to(device)

            # 构造 padding mask（True 表示是 pad 的位置）
            enc_pad_mask = (enc_input == vocab['<pad>']).to(device)
            dec_pad_mask = (dec_input == vocab['<pad>']).to(device)

            # 前向传播
            logits = model(
                enc_input, 
                dec_input, 
                tgt_mask=tgt_mask,
                enc_pad_mask=enc_pad_mask,
                dec_pad_mask=dec_pad_mask
            )



            # 计算损失（flatten 两个 tensor）
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # [B*T, V]
                targets.view(-1)                   # [B*T]
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            global_step += 1

            # 更新进度条 & TensorBoard
            tpbar.set_postfix(loss=loss.item())
            writer.add_scalar('train/loss_step', loss.item(), global_step)

        # 每轮记录一次平均 loss
        avg_loss = epoch_loss / len(dl)
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}")

    # 关闭 TensorBoard 写入器
    writer.close()

    # 保存模型参数
    torch.save(model.state_dict(), '/mnt/data_1/zfy/self/homework/seq2seq_state.bin')

