import pickle
import torch
import json
from torch.utils.data import DataLoader
from data_fix import get_proc
from model import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  

if __name__ == '__main__':
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # TensorBoard 写入器
    writer = SummaryWriter(log_dir=r'/mnt/data_1/zfy/4/week8/资料/homework/homework2/runs/seq2seq_experiment') 

    # 加载训练数据
    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework2/vocab.bin','rb') as f:
        vocab = pickle.load(f)

    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework2/encoder.json') as f:
        enc_data = json.load(f)

    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework2/decoder.json') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data, dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=get_proc(vocab))

    # 构建训练模型
    model = Seq2Seq(
        enc_emb_size=len(vocab),
        dec_emb_size=len(vocab),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )


        
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    global_step = 0  

    for epoch in range(20):
        model.train()
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            logits = model(enc_input, dec_input)

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

            # 记录loss到TensorBoard
            writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1

    # 保存模型
    torch.save(model.state_dict(), r'/mnt/data_1/zfy/4/week8/资料/homework/homework2/seq2seq_state.bin')

    writer.close()  # 


#