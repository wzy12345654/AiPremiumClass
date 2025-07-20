import pickle
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from process import get_proc
# from EncoderDecoderModel import Seq2Seq 
from EncoderDecoderAttenModel import Seq2Seq 
import torch.optim as optim
from tqdm import tqdm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data 加载训练数据
    with open("../data/vocab.bin", "rb") as f:
        evoc,dvoc = pickle.load(f)
    with open("../data/encoder.json", "r") as f:
        enc_data = json.load(f)
    with open("../data/decoder.json", "r") as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data, dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True,
    collate_fn =  get_proc(evoc,dvoc))
    # load model
    
    # 构建训练模型
    model = Seq2Seq(enc_emb_size=len(evoc), dec_emb_size=len(dvoc), 
                    emb_dim=100, hidden_size=120, dropout=0.5)
    
    model.to(device)
    
    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(25):
        model.train()
        tqbar = tqdm(dl)
        for i, (enc, dec,target) in enumerate(tqbar):
            enc, dec,target = enc.to(device), dec.to(device),target.to(device)
            optimizer.zero_grad()
            logits,_ = model(enc, dec)
            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.shape[-1]), target.view(-1))
            
            loss.backward()
            optimizer.step()
            tqbar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    # 保存模型
    torch.save(model.state_dict(), "../data/seq2seq_state.bin")
        
    
