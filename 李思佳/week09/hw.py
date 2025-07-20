import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import math
import jieba

# 位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放指数值
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 位置编码索引 (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)
        # 添加和batch对应维度 (1, 5000, emb_size)
        pos_embdding = pos_embdding.unsqueeze(-2)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)  # token长度
        # (1, token_len, emb_size)
        add_emb = self.pos_embedding[:token_len, :] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTransformer(nn.Module):

    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
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
        # decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, src_mask,tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        print(f"!!!!{enc_emb.shape} ,{dec_emb.shape} ,{src_mask.shape},  {tgt_mask.shape}, {enc_pad_mask.shape}, {dec_pad_mask.shape}")

        outs = self.transformer(src=enc_emb, tgt=dec_emb, src_mask=src_mask,
                                tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 推理
        return self.predict(outs)
    
    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)
 
class Vocabulary:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, corpus):
        # 字典构建（字符为token、词汇为token）
        no_repeat_tokens = set()
        for cmt in jieba.lcut(corpus):
            no_repeat_tokens.update(list(cmt))  # token list
        # set转换为list，第0个位置添加统一特殊token
        tokens = ['PAD','UNK','<s>','</s>'] + list(no_repeat_tokens)

        vocab = { tk:i for i, tk in enumerate(tokens)}

        return cls(vocab)

def create_mask(src, tgt, pad_idx):
    # 源序列填充掩码 [batch_size, src_seq_len]
    enc_pad_mask = (src == pad_idx)
    
    # 目标序列填充掩码 [batch_size, tgt_seq_len]
    dec_pad_mask = (tgt == pad_idx)
    
    # 源序列自注意力掩码 [src_seq_len, src_seq_len]
    src_len = src.size(1)
    src_mask = torch.zeros((src_len, src_len), device=src.device).type(torch.bool)
    
    # 目标序列自注意力掩码 [tgt_seq_len, tgt_seq_len]
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device) == 1).transpose(0, 1)
    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
    
    return src_mask, tgt_mask, enc_pad_mask, dec_pad_mask


def train_epoch(model, data_loader, optimizer, loss_fn, pad_idx, device):
    model.train()
    total_loss = 0

    for enc_input, dec_input, targets in data_loader:
        # enc_input, dec_input, targets = enc_input.to(device), dec_input.to(device), targets.to(device)
        src_mask, tgt_mask, enc_pad_mask, dec_pad_mask = create_mask(enc_input, dec_input, pad_idx)

        print(f"src_mask shape: {src_mask.shape}, tgt_pad_mask shape: {dec_pad_mask.shape}")


        optimizer.zero_grad()

        output = model(enc_input, dec_input, src_mask,tgt_mask,  enc_pad_mask, dec_pad_mask)

        loss = loss_fn(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)



def get_proc(vocab):
    def batch_proc(data):
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in data:
            enc_idx = [vocab['<s>']] + [vocab.get(tk, vocab['UNK']) for tk in enc] + [vocab['</s>']]
            dec_idx = [vocab['<s>']] + [vocab.get(tk, vocab['UNK']) for tk in dec] + [vocab['</s>']]
            
            enc_ids.append(torch.tensor(enc_idx))
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            labels.append(torch.tensor(dec_idx[1:]))
        
        # 计算最大长度（统一使用相同的最大长度）
        max_len = max(max(len(seq) for seq in enc_ids), 
                     max(len(seq) for seq in dec_ids))
        
        # 统一填充到相同长度
        enc_input = pad_sequence(enc_ids, batch_first=True, padding_value=vocab['PAD'])
        dec_input = pad_sequence(dec_ids, batch_first=True, padding_value=vocab['PAD'])
        targets = pad_sequence(labels, batch_first=True, padding_value=vocab['PAD'])
        
        # 确保所有序列长度相同
        if enc_input.size(1) < max_len:
            padding = torch.full((enc_input.size(0), max_len - enc_input.size(1)), 
                              vocab['PAD'], dtype=torch.long)
            enc_input = torch.cat([enc_input, padding], dim=1)
        
        if dec_input.size(1) < max_len:
            padding = torch.full((dec_input.size(0), max_len - dec_input.size(1)), 
                              vocab['PAD'], dtype=torch.long)
            dec_input = torch.cat([dec_input, padding], dim=1)
        
        if targets.size(1) < max_len:
            padding = torch.full((targets.size(0), max_len - targets.size(1)), 
                              vocab['PAD'], dtype=torch.long)
            targets = torch.cat([targets, padding], dim=1)
        
        return enc_input, dec_input, targets
    return batch_proc

if __name__ == '__main__':

    # 参数设置
    d_model = 128
    nhead = 8
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 512
    dropout = 0.1
    batch_size = 1
    num_epochs = 10
    pad_idx = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    enc_vocab = Vocabulary.from_documents(corpus)
    dec_vocab = Vocabulary.from_documents(corpus)

    print(enc_vocab.vocab)
    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    print(enc_tokens)
    print(dec_tokens)
    
    data = list(zip(enc_tokens, dec_tokens))
    data_loader = DataLoader(data, batch_size=batch_size, collate_fn=get_proc(enc_vocab.vocab))

    # 模型、优化器和损失函数
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout,
                               len(enc_vocab.vocab), len(dec_vocab.vocab)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 训练循环
    for epoch in range(num_epochs):
        loss = train_epoch(model, data_loader, optimizer, loss_fn, pad_idx, device)
        print(f"Epoch {epoch+1}, Loss: {loss}")

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

