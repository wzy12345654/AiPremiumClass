import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # 行缩放比
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000)/ emb_size)
        # 位置编码索引(5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen,1)
        # 编码矩阵 (5000, emb_size)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 添加和batch对应维度 (1，5000,emb_size)
        pos_embedding = pos_embedding.unsqueeze(0)
        # print(pos_embedding.shape)
        self.dropout = nn.Dropout(dropout)
        # 当前矩阵不参与参数更新
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        token_len = token_embedding.size(1) # token_embedding的长度
        # (1, token_len, emb_size)截取一部分长度
        # print("self.pos_embedding[:token_len, :].shape",self.pos_embedding[:token_len, :].shape)
        # print("token_embedding.shape",token_embedding.shape)
        add_emb = self.pos_embedding[:,:token_len, :] + token_embedding
        return self.dropout(add_emb)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self,d_model, n_head,num_enc_layers, num_dec_layers, 
                 dim_feedforward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_head,
                                          num_decoder_layers=num_dec_layers,
                                          num_encoder_layers=num_enc_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
        # encoder/decoder input embedding
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict
        self.predict = nn.Linear(d_model,dec_voc_size) # token预测基于解码器词典
        # positional encodeing
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算， tgt_mask是三角矩阵
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=enc_pad_mask,
                                tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)
    # 预测
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)

    def deocer(self, dec_inp):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb)

# 1. 构建词典
def build_vocab(tokens):
    # 添加特殊token
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    # 添加其他token
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# 创建数据集类
class TranslationDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab
        
    def __len__(self):
        return len(self.enc_tokens)
    
    def __getitem__(self, idx):
        # 将token转换为索引
        enc_indices = [self.enc_vocab[token] for token in self.enc_tokens[idx]]
        dec_indices = [self.dec_vocab[token] for token in self.dec_tokens[idx]]
        return torch.tensor(enc_indices), torch.tensor(dec_indices)

# 自定义collate函数
def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    
    # 3.1 填充encoder序列
    enc_padded = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    
    # 3.2 填充decoder序列
    dec_padded = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    
    # 3.2 和 3.3 创建decoder的输入和目标
    dec_input = dec_padded[:, :-1]  # 去掉最后一个token
    dec_target = dec_padded[:, 1:]  # 去掉第一个token
    
    return enc_padded, dec_input, dec_target

# 创建mask
def create_masks(enc_input, dec_input):
    # 4.1 创建decoder的mask（上三角矩阵）
    seq_len = dec_input.size(1)
    dec_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    
    # 4.2 创建encoder的padding mask
    enc_pad_mask = (enc_input == 0)
    
    # 4.3 创建decoder的padding mask
    dec_pad_mask = (dec_input == 0)
    
    return dec_mask, enc_pad_mask, dec_pad_mask

# 主函数
def main():
    # 构建词典
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    all_tokens = set()
    for enc, dec in zip(enc_tokens, dec_tokens):
        all_tokens.update(enc)
        all_tokens.update(dec)
    
    enc_vocab = build_vocab(all_tokens)
    dec_vocab = build_vocab(all_tokens)
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    
    # 5. 创建模型
    model = Seq2SeqTransformer(
        d_model=512,          
        n_head=8,            
        num_enc_layers=6,   
        num_dec_layers=6,    
        dim_feedforward=2048,
        dropout=0.1,         
        enc_voc_size=len(enc_vocab),
        dec_voc_size=len(dec_vocab)
    )
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    
    # 6. 训练模型
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for enc_input, dec_input, dec_target in dataloader:
            # 将数据移到设备上
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_target = dec_target.to(device)
            
            # 创建mask
            dec_mask, enc_pad_mask, dec_pad_mask = create_masks(enc_input, dec_input)
            dec_mask = dec_mask.to(device)
            
            # 前向传播
            output = model(enc_input, dec_input, dec_mask, enc_pad_mask, dec_pad_mask)
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), dec_target.view(-1))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    vocab_data = {
        'enc_vocab': enc_vocab,
        'dec_vocab': dec_vocab
    }
    torch.save(vocab_data, 'vocab_data.pth')

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     # pos_emb = PositionalEncoding(10, .5)
#     # input_emb = torch.zeros((15,10))
#     # result = pos_emb(input_emb)
#     # print(result.shape)
#     corpus= "人生得意须尽欢，莫使金樽空对月"
#     chs = list(corpus)
    
#     enc_tokens, dec_tokens = [],[]

#     for i in range(1,len(chs)):
#         enc = chs[:i]
#         dec = ['<s>'] + chs[i:] + ['</s>']
#         enc_tokens.append(enc)
#         dec_tokens.append(dec)
    
