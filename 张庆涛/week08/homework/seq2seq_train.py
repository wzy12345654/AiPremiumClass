import pickle
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from process import get_proc, Vocabulary
# from EncoderDecoderModel import Seq2Seq 
from EncoderDecoderAttenModel import Seq2Seq 
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    writer = SummaryWriter()
    train_loss_cnt = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data 加载训练数据
    vocab_file = '../data/couplet/vocabs'
    vocab = Vocabulary.from_documents(vocab_file)
    
    with open('../data/couplet/encoder.json') as f:
        enc_data = json.load(f)
    with open('../data/couplet/decoder.json') as f:
        dec_data = json.load(f)


    ds = list(zip(enc_data, dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True,
    collate_fn =  get_proc(vocab.vocab))
    # load model
    
    # 构建训练模型
    model = Seq2Seq(enc_emb_size=len(vocab.vocab), dec_emb_size=len(vocab.vocab), 
                    emb_dim=200, hidden_size=250, dropout=0.5,
                    state_type='add'
                    )
    
    model.to(device)
    
    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(20):
        model.train()
        tqbar = tqdm(dl)
        for enc_input, dec_input, targets in tqbar:
            enc_input, dec_input,targets = enc_input.to(device), dec_input.to(device),targets.to(device)
            optimizer.zero_grad()
            logits,_ = model(enc_input, dec_input )
            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            tqbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
            train_loss_cnt += 1
            
    # 保存模型
    torch.save(model.state_dict(), "../data/couplet/seq2seq_state_add.bin")
        
    
