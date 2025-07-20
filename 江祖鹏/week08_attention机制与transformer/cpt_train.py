from os import write
import pickle
from tkinter import W
import torch
import json
from torch.utils.data import DataLoader
from cpt_process import get_proc, Vocabulary
from cpt_EncDecAttenModel import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    Writer = SummaryWriter()
    train_loss_cnt = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    vocab_file = 'couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)

    with open('couplet/encoder.json') as f:
        enc_data = json.load(f)
    with open('couplet/decoder.json') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data, dec_data))
    dl = DataLoader(ds, batch_size=512, shuffle=True, collate_fn=get_proc(vocab.vocab))

    # 构建训练模型
    # 模型构建
    model = Seq2Seq(
        enc_emb_size=len(vocab.vocab),
        dec_emb_size=len(vocab.vocab),
        emb_dim=200,
        hidden_size=250,
        dropout=0.5,
    )
    model.to(device)

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(1):
        model.train()
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播
            logits, _ = model(enc_input, dec_input)

            #计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tpbar.set_description(f"Epoch {epoch+1}/{20} Loss: {loss.item():.4f}")
            Writer.add_scalar('train_loss', loss.item(), train_loss_cnt)
            train_loss_cnt += loss.item()
        
    # 保存模型
    torch.save(model.state_dict(), 'couplet/cpt_model2.bin')


            


