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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformer_model import Seq2SeqTransformer

# 构建词典
def build_vocab(token_lists):
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    idx =3
    for tokens in token_lists:
      for t in tokens:
        if t not in vocab:
          vocab[t] = idx
          idx += 1
    return vocab

class MyDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        # 初始化函数，传入编码器输入、解码器输入、编码器词汇表和解码器词汇表
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def __len__(self):
        return len(self.enc_tokens)

    def __getitem__(self, idx):
      enc = [self.enc_vocab[t] for t in self.enc_tokens[idx]] # [1,2,3,4,5]
      dec = [self.dec_vocab[t] for t in self.dec_tokens[idx]] # [1,2,3,4,5]
      # enc = [1,2,3,4,5] dec = [1,2,3,4,5]
      return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)
        
def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    enc_batch = pad_sequence(enc_batch,batch_first=True, padding_value=0)
    
    # decoder 输入 需要去掉最后一个
    dec_input = [dec[:-1] for dec in dec_batch]
    # decoder 输出 需要去掉第一个
    dec_output = [dec[1:] for dec in dec_batch]
    dec_input = pad_sequence(dec_input,batch_first=True, padding_value=0)
    dec_output = pad_sequence(dec_output,batch_first=True, padding_value=0)
    return enc_batch, dec_input, dec_output
    
    # 构建mask
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask
if __name__ == '__main__':
    
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
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
    inv_dec_vocab = { v:k for k,v in dec_vocab.items()}    
    # 构建数据集和dataloader
    dataset = MyDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=2,shuffle=True,collate_fn=collate_fn)
    # 模型参数
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead,num_enc_layers,num_dec_layers,dim_forward,dropout,enc_voc_size,dec_voc_size).to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练
    for epoch in range(200):
        model.train()
        total_loss = 0
        for enc_batch, dec_input, dec_output in dataloader:
            enc_batch, dec_input, dec_output = enc_batch.to(device), dec_input.to(device), dec_output.to(device)
          
            tgt_mask = generate_square_subsequent_mask(dec_input.size(1)).to(device)
            enc_pad_mask = (enc_batch == 0)
            dec_pad_mask = (dec_input == 0)
            logits = model(enc_batch, dec_input, tgt_mask, enc_pad_mask, dec_pad_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_output.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(dataloader)}')
        
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    
    # 推理
    