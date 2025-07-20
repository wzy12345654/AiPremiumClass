import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from process import get_proc
from EncodeDecodeAttenModel import Seq2Seq
import torch.optim as optim
from tqdm import tqdm

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取数据
    with open("encoder.json", "r", encoding="utf-8") as f:
        enc_data = json.load(f)

    with open("decoder.json", "r", encoding="utf-8") as f:
        dec_data = json.load(f)

    # 读取词典
    with open("vocab.pkl", "rb") as f:
        enc_vocab, dec_vocab = pickle.load(f)
    
    dataset = list(zip(enc_data, dec_data))

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=get_proc(enc_vocab, dec_vocab))

    # 构建seq2seq模型
    # 定义超参数
    enc_input_dim = len(enc_vocab)
    dec_input_dim = len(dec_vocab)
    emb_dim = 128
    hidden_dim = 128
    dropout = 0.5

    model = Seq2Seq(enc_input_dim, dec_input_dim, emb_dim, hidden_dim, dropout)
    model.to(device)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tpbar = tqdm(dataloader)
        for enc_input, dec_input, labels in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            labels = labels.to(device)

            outputs, _ = model(enc_input, dec_input)

            # 计算损失
            # 展平为[batch_size * seq_len, vocab_size]
            # labels: [batch_size, seq_len] -> [batch_size * seq_len]
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tpbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "seq2seq_model.pth")