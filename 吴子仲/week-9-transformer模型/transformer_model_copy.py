import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim, dropout, max_len=5000):
        super().__init__()

        # 变化率参数权重
        theta = torch.exp(- torch.arange(0, emb_dim, step=2) * math.log(10000.0) / emb_dim)
        # 位置索引张量
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        # 初始化位置权重矩阵
        pos_emb = torch.zeros(max_len, emb_dim)
        # 奇数行，偶数行词向量值分别对应正弦余弦
        pos_emb[:, 0::2] = torch.sin(pos * theta)
        pos_emb[:, 1::2] = torch.cos(pos * theta)
        pos_emb = pos_emb.unsqueeze(0) #和token embedding对齐
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 住处当前矩阵不参与参数更新
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, token_emb):
        # token embedding + position embedding
        emb = token_emb + self.pos_emb[:, :token_emb.size(1)]
        return self.dropout(emb)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,
                 enc_vocab_size, dec_vocab_size):
        super().__init__()

        # transformer
        self.transformer = nn.Transformer(
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            batch_first=True
        )

        # encoder input embedding
        self.enc_emb = nn.Embedding(enc_vocab_size, d_model)
        # decoder input embedding
        self.dec_emb = nn.Embedding(dec_vocab_size, d_model)
        # predictor
        self.predictor = nn.Linear(d_model, dec_vocab_size)
        # posiitonal encoding
        self.pos_enc = PositionalEncoding(d_model, dropout)

    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask):
        # multi-head attention 基于位置编码embedding生成
        src_emb = self.pos_enc(self.enc_emb(src))
        tgt_emb = self.pos_enc(self.dec_emb(tgt))
        # transformer
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_pad_mask,
                                  tgt_key_padding_mask=tgt_pad_mask)
        # 推理
        return self.predictor(output)
    
    def encoder(self, enc_input):
        enc_emb = self.pos_enc(self.enc_emb(enc_input))
        return self.transformer.encoder(enc_emb)
    
    def decoder(self, dec_input, enc_output, tgt_mask):
        dec_emb = self.pos_enc(self.dec_emb(dec_input))
        return self.transformer.decoder(dec_emb, enc_output, tgt_mask=tgt_mask)

class Vocabulary:
    """
    词汇表类
    """
    def __init__(self, vocab):
        self.vocab = vocab
    
    # 根据数据构建词汇表
    # 这里的data是读取文件后的list集合
    @classmethod
    def build_from_data(cls, data):
        vocab_set = set()
        for item in data:
            # 处理字符列表或单词列表
            if isinstance(item, list):
                vocab_set.update(item)
            else:
                vocab_set.update(list(item))
        vocab_set = ['PAD', 'UNK', '<s>', '</s>'] + list(vocab_set)  # PAD:padding, UNK: unknown
        # 构建词汇到索引的映射
        vocab = {word: i for i, word in enumerate(vocab_set)}
        return cls(vocab)

def get_proc(vocab):
    def batch_proc(data):
        """
        批次数据处理函数
        """
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in data:
            # token -> index 首尾添加起始结束符
            enc_idx = [vocab["<s>"]] + [vocab.get(word, vocab['UNK']) for word in enc] + [vocab["</s>"]]
            dec_idx = [vocab["<s>"]] + [vocab.get(word, vocab['UNK']) for word in dec] + [vocab["</s>"]]

            enc_ids.append(torch.tensor(enc_idx, dtype=torch.long))
            dec_ids.append(torch.tensor(dec_idx[:-1], dtype=torch.long))  # 去掉最后一个</s>
            labels.append(torch.tensor(dec_idx[1:], dtype=torch.long))  # 去掉第一个<s>

        # padding
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        # 返回填充后的数据
        return enc_input, dec_input, labels
    # 返回回调函数
    return batch_proc

# 创建mask
def create_mask(enc_input, dec_input):
    dec_sz = dec_input.size(1)
    # 生成方阵
    mask = torch.ones(dec_sz, dec_sz)
    # 下三角截取
    mask = torch.tril(mask)
    # 条件填充
    mask = mask.masked_fill(mask==0, value=float('-inf'))
    mask = mask.masked_fill(mask==1, value=float(0.0))

    # enc_pad_mask, dec_pad_mask
    enc_pad_mask = enc_input == 0   # 假设 0 是 padding 的标记
    dec_pad_mask = dec_input == 0   # 假设 0 是 padding 的标记
    return mask, enc_pad_mask, dec_pad_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, vocab):
    """使用贪心搜索进行解码"""
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # 编码源序列
    memory = model.encoder(src)
    
    # 初始化解码器输入
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    # 词汇表的逆映射
    idx2word = {i: w for w, i in vocab.items()}
    
    for i in range(max_len-1):
        memory = memory.to(device)
        
        # 创建目标序列的mask
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to(device)
        
        # 解码
        out = model.decoder(ys, memory, tgt_mask)
        prob = model.predictor(out[:, -1])
        
        # 获取概率最大的词
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # 添加到输出序列
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        # 如果是结束符，则停止生成
        if next_word == vocab['</s>']:
            break
    
    # 将索引转换为词
    decoded_words = []
    for idx in ys[0].tolist():
        word = idx2word.get(idx, 'UNK')
        decoded_words.append(word)
        if word == '</s>':
            break
    
    return decoded_words[1:-1]  # 去掉<s>和</s>

def generate_square_subsequent_mask(sz):
    """生成上三角全为-inf，下三角全为0的mask矩阵"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    corpus = "天生我材必有用，千金散尽还复来。"
    chs = list(corpus)

    enc_tokens, dec_tokens = [], []

    for i in range(1, len(chs)):
        enc_tokens.append(chs[:i])
        dec_tokens.append(chs[i:])  # 修正：不需要手动添加<s>和</s>，由batch_proc处理
    
    # 构建词典（修正：使用完整的训练数据）
    vocab = Vocabulary.build_from_data(enc_tokens + dec_tokens)
    vocab_size = len(vocab.vocab)
    print(f"词汇表大小: {vocab_size}")

    # 编码+解码
    dataset = list(zip(enc_tokens, dec_tokens))

    dataloader = DataLoader(
        dataset,
        batch_size = 2,
        shuffle = True,
        collate_fn = get_proc(vocab.vocab)
    )

    # 定义超参数（调整为更合适的值）
    d_model = 64
    nhead = 4  # 修正：64不能被8整除，改为4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = d_model * 4
    dropout = 0.1  # 修正：减小dropout值，提高训练稳定性

    # 修正：使用正确的词汇表大小
    model = Seq2SeqTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                               dim_feedforward, dropout, vocab_size, vocab_size)
    model.to(device=device)

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD索引的损失

    # 训练模型
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tpbar = tqdm(dataloader)
        for enc_input, dec_input, labels in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            labels = labels.to(device)

            # 生成mask
            dec_mask, enc_pad_mask, dec_pad_mask = create_mask(enc_input, dec_input)
            dec_mask = dec_mask.to(device)
            enc_pad_mask = enc_pad_mask.to(device)
            dec_pad_mask = dec_pad_mask.to(device)

            # 修正：直接使用模型输出，不需要解包
            outputs = model(enc_input, dec_input, dec_mask, enc_pad_mask, dec_pad_mask)

            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tpbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "seq2seq_transformer_model.pth")
    print("模型训练完成并保存")

    # 测试模型
    model.eval()
    
    # 选择一个样本进行测试
    test_enc, test_dec = dataset[0]
    print(f"源序列: {''.join(test_enc)}")
    print(f"目标序列: {''.join(test_dec)}")
    
    # 准备输入
    test_enc_idx = [vocab.vocab["<s>"]] + [vocab.vocab.get(word, vocab.vocab['UNK']) for word in test_enc] + [vocab.vocab["</s>"]]
    test_enc_tensor = torch.tensor(test_enc_idx, dtype=torch.long).unsqueeze(0).to(device)
    
    # 创建源序列的mask
    test_enc_mask = torch.zeros((1, test_enc_tensor.size(1))).type(torch.bool).to(device)
    
    # 解码
    output = greedy_decode(model, test_enc_tensor, test_enc_mask, max_len=50, 
                          start_symbol=vocab.vocab["<s>"], vocab=vocab.vocab)
    
    print(f"生成序列: {''.join(output)}")