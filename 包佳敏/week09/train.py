import torch
import json
from torch.utils.data import DataLoader
from process import get_proc, Vocabulary
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#位置编码矩阵
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #位置编码矩阵 (5000, emb_size)
        pos_embdding = torch.zeros(max_len, emb_size)
        #位置编码索引（5000，1）
        position = torch.arange(0, max_len, dtype=torch.float).reshape(max_len, 1)  # (max_len, 1)
        #行缩放指数值
        den = torch.exp(torch.arange(0, emb_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / emb_size))
        pos_embdding[:, 0::2] = torch.sin(position * den) #奇数列
        pos_embdding[:, 1::2] = torch.cos(position * den) #偶数列
        #添加和batch对应纬度（1，5000， emb_size）
        pos_embdding = pos_embdding.unsqueeze(0)  # (1, max_len, d_model)
        #dropput
        self.dropout = nn.Dropout(dropout)
        #注册当前矩阵不参与梯度更新,注册到缓冲区
        self.register_buffer('pos_embdding', pos_embdding)

    #前向传播,让系统回调
    def forward(self, token_embdding):
        #token_embdding: (batch_size, seq_len, emb_size)    
        token_len = token_embdding.size(1) #token长度
        #pos_embdding: (1, token_len, emb_size)
        add_emb= self.pos_embdding[:, :token_len, :] + token_embdding
        return self.dropout(add_emb)

class Seq2SeqTrans(nn.Module):
    def __init__(self, input_dim, output_dim, emb_size, n_heads, num_layers, dropout=0.1):
        super(Seq2SeqTrans, self).__init__()
        # input_dim: 输入词汇表大小
        # output_dim: 输出词汇表大小
        # emb_size: 词向量维度
        self.enc_embedding = nn.Embedding(input_dim, emb_size)
        self.dec_embedding = nn.Embedding(output_dim, emb_size)
        #token预测基于解码器词典
        self.predict = nn.Linear(emb_size, output_dim)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        # Transformer模型 d_model词向量大小，nhead头的数量，num_encoder_layers和num_decoder_layers分别表示编码器和解码器的堆叠层数
        self.transformer = nn.Transformer(d_model=emb_size, nhead=n_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward = 4 * emb_size,
                                          dropout=dropout,batch_first=True)

    def forward(self, enc_inp, dec_inp, tgt_mask,tgt_key_padding_mask,src_key_padding_mask):
        #multi head attention之前基于位置编码embedding生成
        enc_emb = self.positional_encoding(self.enc_embedding(enc_inp))
        dec_emb = self.positional_encoding(self.dec_embedding(dec_inp))
        #调用transformer计算
        output = self.transformer(enc_emb, dec_emb, tgt_mask, 
                                  src_key_padding_mask=tgt_key_padding_mask,
                                  tgt_key_padding_mask=src_key_padding_mask)
        #推理
        return self.predict(output)  
    
    #推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.positional_encoding(self.enc_embedding(enc_inp))
        return self.transformer.encoder(enc_emb)
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.positional_encoding(self.dec_embedding(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)       



    #构建encoder和decoder的词典    
    #模型训练数据： X：[enc_token_matrix, dec_token_matrix]  shifted right  
    #模型训练数据： Y：[dec_token_matrix] shifted right
    #1.通过词典把token转化为token_index
    #2.通过Dataloader把encoder,decoder封装为带有batch的训练数据
    #3.Dataloader的collate_fn调用自定义转化方法，填充模型训练数据
    #   3.1 encoder矩阵使用pad_sequence填充
    #   3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
    #   3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
    #4.创建mask
    #   4.1 decoder_mask:上三角填充-inf的mask
    #   4.2 enc_pad_mask:(enc矩阵 == 0)
    #   4.3 dec_pad_mask:(dec矩阵 == 0)
    #5.创建模型（根据GPU内存大小设计编码和解码器的参数和层数）、优化器、损失函数
    #6.训练模型并保存 
if __name__ == '__main__':
    
    writer = SummaryWriter()
    train_loss_cnt = 0

    # 加载训练数据
    vocab_file = '/Users/baojiamin/Desktop/couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)

    with open('/Users/baojiamin/Desktop/couplet/train/encoder.json') as f:
        enc_data = json.load(f)
    with open('/Users/baojiamin/Desktop/couplet/train/decoder.json') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data,dec_data))
    #get_proc是自定义的回调函数,
    # 1.通过词典把token转化为token_index
    # 2.通过Dataloader把encoder,decoder封装为带有batch的训练数据
    # 3.Dataloader的collate_fn调用自定义转化方法，填充模型训练数据
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=get_proc(vocab.vocab)) 
    # 构建训练模型
    # 模型构建
    model = Seq2SeqTrans(
        input_dim=len(vocab.vocab),
        output_dim=len(vocab.vocab),
        emb_size=128,
        n_heads=4,
        num_layers=2
    )

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(40):
        model.train()
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            # 前向传播 
            logits= model(enc_input, dec_input, None, None, None)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            writer.add_scalar('Loss/train', loss.item(), train_loss_cnt)
            train_loss_cnt += 1

    torch.save(model.state_dict(), 'seq2seqtrans_state.bin')