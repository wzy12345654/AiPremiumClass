import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

#位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        #行缩放指数值
        den = torch.exp(torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        #位置编码索引 (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen,1)
        #编码矩阵 (5000, emb_size)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        #添加和batch对应维度 (1, 5000, emb_size)
        pos_embedding = pos_embedding.unsqueeze(0)
        #dropout
        self.dropout = nn.Dropout(dropout)
        #注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        token_len = token_embedding.size(1) #token长度
        #(1, token_len, emb_size)
        add_emb = self.pos_embedding[:,:token_len, :] + token_embedding
        return self.dropout(add_emb)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        # transformer
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder input embedding
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        #decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size) # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        #multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        #调用transformer
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask,
                         tgt_key_padding_mask=dec_pad_mask)
        #推理
        return self.predict(outs)

    #推理环节使用方法
    def encoder(self, enc_inp):
        emc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(emc_emb)
    
    def decoder(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb,memory, dec_mask)

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self,enc_seqs, dec_seqs):
        self.enc_seqs = enc_seqs
        self.dec_seqs = dec_seqs

    def __len__(self):
        return len(self.enc_seqs)
    
    def __getitem__(self, idx):
        return self.enc_seqs[idx], self.dec_seqs[idx]
    
def collate_fn(vocab):

    def batch_proc(data):
        enc_inputs, dec_inputs, lables = [], [], []
        for enc, dec in data:
            enc_ids = [token_idx[token]for token in enc]
            dec_input = dec[:-1]
            label = dec[1:]
            dec_ids = [token_idx[token] for token in dec_input]
            target_ids = [token_idx[token] for token in label]

            enc_inputs.append(torch.tensor(enc_ids))
            dec_inputs.append(torch.tensor(dec_ids))
            lables.append(torch.tensor(target_ids))

        #填充
        enc_inputs = pad_sequence(enc_inputs, batch_first=True, padding_value=token_idx['<pad>'])
        dec_inputs = pad_sequence(dec_inputs, batch_first=True, padding_value=token_idx['<pad>'])
        labels = pad_sequence(lables, batch_first=True, padding_value=token_idx['<pad>'])

        #填充mask
        enc_pad_mask = (enc_inputs == token_idx['<pad>'])
        dec_pad_mask = (dec_inputs == token_idx['<pad>'])

        #解码器mask
        tgt_len = dec_inputs.size(1)
        tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len)), diagonal=1).bool()

        return enc_inputs, dec_inputs, labels, tgt_mask, enc_pad_mask, dec_pad_mask
    return batch_proc

    

if __name__ == '__main__':

    #模型数据
    #语料：encoder， decoder
    corpus = '人生得意须尽欢，莫使金樽空对月'
    chs = list(corpus)
    print(chs)
    enc_tokens, dec_tokens = [],[]
    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    chars = set()
    for token_list in enc_tokens + dec_tokens:
        chars.update(token_list)

    #构建词汇表
    vocab = ['<pad>', '<s>', '</s>'] + list(chars)
    token_idx = {token:idx for idx, token in enumerate(vocab)}
    vocab_size = len(vocab)
    print(vocab_size)

    #创建数据加载器
    dataset = TranslationDataset(enc_tokens, dec_tokens)
    dataloader = DataLoader(dataset, 
                            batch_size=2, 
                            shuffle=True, 
                            collate_fn=collate_fn(vocab))
    
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #初始化模型
    model = Seq2SeqTransformer(d_model=128,
                               nhead=8,
                               num_enc_layers=6,
                               num_dec_layers=6,
                               dim_forward=512,
                               dropout=0.1,
                               enc_voc_size=vocab_size,
                               dec_voc_size=vocab_size)
    model.to(devices)
    #定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=token_idx['<pad>'])

    #训练模型
    for epoch in range(200):
        total_loss = 0
        model.train()
        for enc_inp, dec_inp, lable, tgt_mask, enc_pad_mask, dec_pad_mask in dataloader:
            enc_inp = enc_inp.to(devices)
            dec_inp = dec_inp.to(devices)
            lable = lable.to(devices)
            tgt_mask = tgt_mask.to(devices)
            enc_pad_mask = enc_pad_mask.to(devices)
            dec_pad_mask = dec_pad_mask.to(devices)

            optimizer.zero_grad()
            outputs = model(enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask)
            loss = criterion(outputs.view(-1, vocab_size), lable.view(-1))
            loss.backward()
            #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{200}], Loss: {avg_loss:.4f}')



