# 构建encoder和docoder的词典

# 模型训练数据： X：([enc_token_matrix], [dec_token_matrix] shifted right)，
# y [dec_token_matrix] shifted

# 1. 通过词典把token转换为token_index
# 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
# 3. Dataloader的collate_fn调用自定义转换方法，填充模型训练数据
#    3.1 encoder矩阵使用pad_sequence填充
#    3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
#    3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
# 4. 创建mask
#    4.1 dec_mask 上三角填充-inf的mask
#    4.2 enc_pad_mask: (enc矩阵 == 0）
#    4.3 dec_pad_mask: (dec矩阵 == 0)
# 5. 创建模型（根据GPU内存大小设计编码和解码器参数和层数）、优化器、损失
# 6. 训练模型并保存

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformer_model import Seq2SeqTransformer

# 1. 构建词典
def build_vocab(token_lists):
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    idx = 3
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab

# 2. 数据集
class MyDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def __len__(self):
        return len(self.enc_tokens)

    def __getitem__(self, idx):
        enc = [self.enc_vocab[t] for t in self.enc_tokens[idx]]
        dec = [self.dec_vocab[t] for t in self.dec_tokens[idx]]
        return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)

# 3. collate_fn
def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    enc_batch = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    
    # dec_in: 去掉最后一个
    dec_in = [dec[:-1] for dec in dec_batch]
    # dec_out: 去掉第一个
    dec_out = [dec[1:] for dec in dec_batch]
    
    dec_in = pad_sequence(dec_in, batch_first=True, padding_value=0)
    # dec_out: 去掉第一个
    dec_out = pad_sequence(dec_out, batch_first=True, padding_value=0)
    return enc_batch, dec_in, dec_out

# 4. mask
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask

# ========== 主流程 ==========
if __name__ == '__main__':
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    # 构建词典
    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)
    inv_dec_vocab = {v: k for k, v in dec_vocab.items()}

    # 构建数据集和dataloader
    dataset = MyDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 模型参数
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size, dec_voc_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 训练
    for epoch in range(50):
        model.train()
        total_loss = 0
        for enc_batch, dec_in, dec_out in dataloader:
            enc_batch, dec_in, dec_out = enc_batch.to(device), dec_in.to(device), dec_out.to(device)
            tgt_mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            enc_pad_mask = (enc_batch == 0)
            dec_pad_mask = (dec_in == 0)
            logits = model(enc_batch, dec_in, tgt_mask, enc_pad_mask, dec_pad_mask)
            # logits: [B, T, V]  dec_out: [B, T]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer.pth')
    print("模型已保存。")