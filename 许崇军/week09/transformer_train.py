import pickle
import torch
import json
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from transformer_model import Seq2SeqTransformer
from process import get_proc, Vocabulary
from transformer_model import generate_square_subsequent_mask

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载训练数据
    with open('vocabs.bin','rb') as f:
        evoc, dvoc = pickle.load(f)

    with open('encoders.json',encoding='utf-8') as f:
        enc_data = json.load(f)
    with open('decoders.json',encoding='utf-8') as f:
        dec_data = json.load(f)

    dataset = list(zip(enc_data,dec_data))
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=get_proc(evoc, dvoc))

    d_model = 256
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 512
    dropout = 0.1


    # 初始化模型
    model = Seq2SeqTransformer(
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        dim_forward=dim_forward,
        dropout=dropout,
        enc_voc_size=len(evoc),
        dec_voc_size=len(dvoc)
    )
    model.to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=dvoc['PAD'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', patience=2, verbose=True)
    # 训练
    for epoch in range(50):  # 示例迭代次数
        model.train()
        total_loss = 0
        for enc_input, dec_input, targets in dataloader:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)
            tgt_mask = generate_square_subsequent_mask(dec_input.size(1)).to(device)
            # 前向传播
            outputs = model(
                enc_inp=enc_input,
                dec_inp=dec_input,
                tgt_mask=tgt_mask,
                src_padding_mask=(enc_input == 0),
                tgt_padding_mask=(dec_input == 0)
            )
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        scheduler.step(avg_loss)
    torch.save(model.state_dict(), 'transformer_final.bin')