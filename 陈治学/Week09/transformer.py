import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 1. 特殊标记定义
PAD_IDX = 0  # 填充标记
BOS_IDX = 1  # 序列开始标记
EOS_IDX = 2  # 序列结束标记
UNK_IDX = 3  # 未知词标记

# 2. 位置编码（Positional Encoding）
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# 3. Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# 4. Seq2Seq Transformer模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, 
                 nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.output_layer = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, 
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.output_layer(outs)

# 5. 掩码生成函数
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).float()
    
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# 6. 自定义数据集（根据文档内容实现）
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        # 添加BOS和EOS标记
        src_tensor = torch.tensor([BOS_IDX] + 
                                 [self.src_vocab.get(word, UNK_IDX) for word in self.src_sentences[idx]] + 
                                 [EOS_IDX])
        
        tgt_tensor = torch.tensor([BOS_IDX] + 
                                 [self.tgt_vocab.get(word, UNK_IDX) for word in self.tgt_sentences[idx]] + 
                                 [EOS_IDX])
        return src_tensor, tgt_tensor

# 7. 数据加载器处理函数
def collate_batch(batch):
    src_batch, tgt_batch = [], []
    for src_item, tgt_item in batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# 8. 训练函数
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]  # 解码器输入（移位的目标序列）
        tgt_output = tgt[1:, :]  # 解码器输出目标
        
        # 创建mask
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device) if src_padding_mask is not None else None
        tgt_padding_mask = tgt_padding_mask.to(device) if tgt_padding_mask is not None else None
        
        # 前向传播
        logits = model(
            src, tgt_input, src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask, src_padding_mask
        )
        
        # 计算损失
        loss = criterion(logits.view(-1, logits.shape[-1]), tgt_output.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 9. 主训练流程
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数设置（根据文档）
    SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小
    TGT_VOCAB_SIZE = 10000  # 目标语言词汇表大小
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 2048
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DROPOUT = 0.1
    BATCH_SIZE = 128
    LR = 0.0001
    EPOCHS = 10
    
    # 创建模型
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    src_sentences = [["I", "love", "transformer", "models"], 
                    ["This", "is", "an", "example"]]
    tgt_sentences = [["J'adore", "les", "modèles", "transformer"], 
                    ["C'est", "un", "exemple"]]
    
    # 创建词汇表（简化版，实际应从数据集中构建）
    src_vocab = {word: i+4 for i, word in enumerate(set([word for sent in src_sentences for word in sent]))}
    tgt_vocab = {word: i+4 for i, word in enumerate(set([word for sent in tgt_sentences for word in sent]))}
    
    # 添加特殊标记
    src_vocab['<pad>'] = PAD_IDX
    src_vocab['<bos>'] = BOS_IDX
    src_vocab['<eos>'] = EOS_IDX
    src_vocab['<unk>'] = UNK_IDX
    
    tgt_vocab['<pad>'] = PAD_IDX
    tgt_vocab['<bos>'] = BOS_IDX
    tgt_vocab['<eos>'] = EOS_IDX
    tgt_vocab['<unk>'] = UNK_IDX
    
    # 创建数据集和数据加载器
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    
    # 训练循环
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model saved to transformer_model.pth")

if __name__ == "__main__":
    main()
