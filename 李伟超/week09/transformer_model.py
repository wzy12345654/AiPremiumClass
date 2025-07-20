import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.nn.utils.rnn import pad_sequence

# 位置编码矩阵
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super().__init__()
        # 创建位置索引和频率缩放因子
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pos = torch.arange(0, maxlen).unsqueeze(1)  # [maxlen, 1]
        
        # 初始化位置编码矩阵
        pos_embedding = torch.zeros((maxlen, emb_size))  # [maxlen, emb_size]
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        # 添加 batch 维度，变成 [1, maxlen, emb_size]，以便 broadcast 给 [batch_size, seq_len, emb_size]
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        # 注册为 buffer，表示它是模型的一部分但不参与训练
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding: [batch_size, seq_len, emb_size]
        seq_len = token_embedding.size(1)
        token_embedding = token_embedding + self.pos_embedding[:, :seq_len, :]
        return self.dropout(token_embedding)

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

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
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
    
 # ========== 构建词典 ==========
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

def build_vocab(tokens_list):
    vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    for tokens in tokens_list:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab[t] for t in tokens]

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def create_pad_mask(matrix, pad_idx=0):
    return matrix == pad_idx

# ========== 构建 Dataset ==========
class CoupletDataset(torch.utils.data.Dataset):
    def __init__(self, enc_data, dec_data):
        self.enc_data = enc_data
        self.dec_data = dec_data

    def __len__(self):
        return len(self.enc_data)

    def __getitem__(self, idx):
        return self.enc_data[idx], self.dec_data[idx]

def collate_fn(batch):
    enc_seqs, dec_seqs = zip(*batch)
    enc_seqs_pad = pad_sequence(enc_seqs, batch_first=True, padding_value=0)
    dec_seqs_pad = pad_sequence(dec_seqs, batch_first=True, padding_value=0)
    dec_input = dec_seqs_pad[:, :-1]
    dec_target = dec_seqs_pad[:, 1:]
    return enc_seqs_pad, dec_input, dec_target

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
    # 构建词典
    all_tokens = enc_tokens + dec_tokens
    vocab = build_vocab(all_tokens)
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    vocab_size = len(vocab)

    # 转换为 id 序列
    enc_ids = [torch.tensor(tokens_to_ids(toks, vocab)) for toks in enc_tokens]
    dec_ids = [torch.tensor(tokens_to_ids(toks, vocab)) for toks in dec_tokens]

    # 构造数据集与加载器
    dataset = CoupletDataset(enc_ids, dec_ids)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 模型构建与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(
        d_model=256,
        nhead=4,
        num_enc_layers=3,
        num_dec_layers=3,
        dim_forward=512,
        dropout=0.1,
        enc_voc_size=vocab_size,
        dec_voc_size=vocab_size
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # ========== 模型训练 ==========
    EPOCHS = 30
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for enc_batch, dec_inp_batch, dec_tar_batch in dataloader:
            enc_batch = enc_batch.to(device)
            dec_inp_batch = dec_inp_batch.to(device)
            dec_tar_batch = dec_tar_batch.to(device)

            # 构建解码器 mask 和 padding mask
            tgt_mask = generate_square_subsequent_mask(dec_inp_batch.size(1)).to(device)
            enc_pad_mask = create_pad_mask(enc_batch).to(device)
            dec_pad_mask = create_pad_mask(dec_inp_batch).to(device)

            # 前向传播
            logits = model(enc_batch, dec_inp_batch, tgt_mask, enc_pad_mask, dec_pad_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), dec_tar_batch.reshape(-1))

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📘 Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # ========== 保存模型 ==========
    torch.save(model.state_dict(), 'seq2seq_couplet_model.pth')
    print("模型已保存为 seq2seq_couplet_model.pth")

    # ========== 下联生成函数 ==========
    def generate(model, enc_input_str, max_len=20):
        model.eval()
        enc_ids = torch.tensor(tokens_to_ids(list(enc_input_str), vocab)).unsqueeze(0).to(device)
        memory = model.encode(enc_ids)
        ys = torch.tensor([[vocab[SOS_TOKEN]]], dtype=torch.long).to(device)

        for i in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = model.predict(out)
            prob = out[:, -1, :]
            next_token = torch.argmax(prob, dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == vocab[EOS_TOKEN]:
                break

        out_tokens = [inv_vocab[idx] for idx in ys[0].tolist()[1:-1]]
        return ''.join(out_tokens)

    # ========== 测试生成 ==========
    test_input = "人生得意"
    generated = generate(model, test_input)
    print(f"\n输入上联: {test_input}")
    print(f"生成下联: {generated}")

