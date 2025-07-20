import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from collections import OrderedDict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 数据集处理
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = [self.src_vocab.get(token, 0) for token in self.src_texts[idx]]
        tgt = [self.tgt_vocab.get(token, 0) for token in self.tgt_texts[idx]]
        return torch.LongTensor(src), torch.LongTensor(tgt)

# Transformer模型
class CustomTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(src_vocab_size, d_model)
        self.embed_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        src = self.embed_src(src) * math.sqrt(self.d_model)
        tgt = self.embed_tgt(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)
        
        output = self.transformer(src, tgt, src_mask, tgt_mask,
                                  src_key_padding_mask, tgt_key_padding_mask,
                                  memory_key_padding_mask)
        return self.fc_out(output)

# 训练流程
def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, tgt[:, :-1])
            output = output.view(-1, output.shape[-1])
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# 主程序
if __name__ == "__main__":
    src_texts = [["hello", "world"], ["how", "are", "you"]]
    tgt_texts = [["bonjour", "le", "monde"], ["comment", "allez-vous"]]
    
    # 构建词汇表
    src_vocab = {"<pad>":0, "<unk>":1, "hello":2, "world":3}
    tgt_vocab = {"<pad>":0, "<unk>":1, "bonjour":2, "le":3, "monde":4, "comment":5, "allez-vous":6}
    
    dataset = TranslationDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CustomTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layer=2
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, device, epochs=5)
    
    # 保存模型
    torch.save(model.state_dict(), "custom_transformer.pth")


def translate_sentence(sentence, model, src_vocab, tgt_vocab, max_len=50):
    model.eval()
    with torch.no_grad():
        # 编码输入句子
        src_indices = [src_vocab.get(token, src_vocab["<unk>"]) for token in sentence.split()]
        src_tensor = torch.LongTensor([src_indices]).to(device)
        
        # 初始化解码器输入
        tgt_indices = [tgt_vocab["<sos>"]]
        for _ in range(max_len):
            # 构造目标序列输入（去掉最后一个token）
            tgt_tensor = torch.LongTensor([tgt_indices]).to(device)
            
            # 获取编码器输出
            memory = model.embed_src(src_tensor) * math.sqrt(model.d_model)
            memory = model.pos_encoder(memory)
            memory = model.transformer.encoder(memory)
            
            # 解码器单步预测
            output = model.embed_tgt(tgt_tensor) * math.sqrt(model.d_model)
            output = model.pos_decoder(output)
            output = model.transformer.decoder(output, memory)
            output = model.fc_out(output[:, -1, :])
            
            # 获取预测token
            prob = output.softmax(dim=-1)
            top1 = prob.argmax().item()
            
            tgt_indices.append(top1)
            
            if top1 == tgt_vocab["<eos>"]:
                break
        
        # 转换为目标语言句子
        translation = [tgt_vocab.get(idx, "<unk>") for idx in tgt_indices]
        return translation

# 使用示例
translated = translate_sentence("hello world", model, src_vocab, tgt_vocab)
print(translated)
