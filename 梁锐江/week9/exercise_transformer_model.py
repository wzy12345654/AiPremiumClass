"""
    self_attn(自注意力)
    作用：处理输入序列自身的注意力(query,key,value都来自同一个输入)
    multihead_attn(交叉注意力)
    作用: 处理两个不同序列之间的注意力

    在位置编码中，相邻两个维度间存在关系 因此用sin和cos捕捉关系
"""
import math

import torch.nn as nn
import torch
import pickle
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, max_length=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        pos_embedding = torch.zeros((max_length, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # shape [1,max_length,embed_size]
        pos_embedding = pos_embedding.unsqueeze(0)

        self.register_buffer('pos_embedding', pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding):
        token_leq = embedding.size(1)

        add_embedding = self.pos_embedding[:, :token_leq, :] + embedding
        return self.dropout(add_embedding)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, enc_vocab_size, dec_vocab_size, dropout):
        super().__init__()
        self.src_embedding = nn.Embedding(enc_vocab_size, d_model)

        self.tgt_embedding = nn.Embedding(dec_vocab_size, d_model)

        self.positional = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)

        self.predict = nn.Linear(d_model, dec_vocab_size)

    def forward(self, src_idxs, trg_idxs):
        src_embed = self.positional(self.src_embedding(src_idxs))
        trg_embed = self.positional(self.tgt_embedding(trg_idxs))
        src_seq_len = src_idxs.size(1)
        src_mask = torch.triu(torch.ones(src_seq_len, src_seq_len) == 1, diagonal=1)

        dec_seq_len = trg_idxs.size(1)
        tgt_mask = torch.triu(torch.ones(dec_seq_len, dec_seq_len) == 1, diagonal=1)

        output = self.transformer(src=src_embed, tgt=trg_embed, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.predict(output)

    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.positional(self.src_embedding(enc_inp))
        return self.transformer.encoder(enc_emb)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.positional(self.tgt_embedding(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)


class BatchProcessor:
    def __init__(self, enc_vocab, dec_vocab):
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def get_word_idx(self, batch_data):
        enc_inputs, dec_inputs, targets = [], [], []
        for enc_doc, dec_doc in batch_data:
            enc_idx = [self.enc_vocab.get(token, 0) for token in enc_doc]
            dec_idx = [self.dec_vocab.get(token, 0) for token in dec_doc]

            enc_inputs.append(torch.tensor(enc_idx))
            dec_inputs.append(torch.tensor(dec_idx[:-1]))
            targets.append(torch.tensor(dec_idx[1:]))

        enc_inputs = rnn.pad_sequence(enc_inputs, batch_first=True)
        dec_inputs = rnn.pad_sequence(dec_inputs, batch_first=True)
        targets = rnn.pad_sequence(targets, batch_first=True)

        return enc_inputs, dec_inputs, targets


if __name__ == '__main__':
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
    with open('F:/githubProject/AiPremiumClass/梁锐江/week8/couple_vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    with open('F:/githubProject/AiPremiumClass/梁锐江/week8/couple_encode.json', 'r', encoding='utf-8') as f:
        enc_docs = json.load(f)

    with open('F:/githubProject/AiPremiumClass/梁锐江/week8/couple_decode.json', 'r', encoding='utf-8') as f:
        dec_docs = json.load(f)

    dataset = list(zip(enc_docs, dec_docs))

    batch_processor = BatchProcessor(enc_vocab, dec_vocab)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=batch_processor.get_word_idx)

    d_model = 512
    epochs = 3
    model = Seq2SeqTransformer(d_model, len(enc_vocab), len(dec_vocab), 0.5)
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    total_step = 0
    model.train()
    for epoch in range(epochs):
        for enc_idxs, dec_idxs, tgt_idxs in train_loader:
            enc_idxs = enc_idxs.to('cuda')
            dec_idxs = dec_idxs.to('cuda')
            tgt_idxs = tgt_idxs.to('cuda')

            optimizer.zero_grad()
            predict = model(enc_idxs, dec_idxs)
            loss = loss_fn(predict.view(-1, len(dec_vocab)), tgt_idxs.view(-1))
            loss.backward()
            optimizer.step()

            total_step += 1
            writer.add_scalar('loss', loss.item(), total_step)
        print(f"epoch running {epoch}")

    writer.close()
    torch.save(model.state_dict(), 'transfomer_model.pth')
