#  3. 编写并实现seq2seq attention版的推理实现。

"""
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pickle
import torch
import torch.nn as nn



# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim,dropout=dropout, 
                          batch_first=True, bidirectional=True)


    def forward(self, token_seq, hidden_type):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        # 返回最后一个时间步的隐藏状态(拼接), 所有时间步的输出（attention准备）
        if hidden_type == 'cat':
            hidden_state = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)
        elif hidden_type == 'sum':
            # 使用线性层调整维度
            hidden_state = (hidden[0] + hidden[1]).unsqueeze(0)
        elif hidden_type == 'mul':
            hidden_state = (hidden[0] * hidden[1]).unsqueeze(0)
        else:
            hidden_state = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)
        
        return hidden_state, outputs

        # 返回最后一个时间步的隐状态（相加）
        # return hidden.sum(dim=0)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_output, dec_output):
        # a_t = h_t @ h_s  
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 1.计算 结合解码token和编码token，关联的权重
        a_t = torch.softmax(a_t, dim=1)
        # 2.计算 关联权重和编码token 贡献值
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=dropout,
                          batch_first=True)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # 解码词典中词汇概率
        # attention层
        self.atteniton = Attention()
        # attention结果转换线性层
        self.atteniton_fc = nn.Linear(hidden_dim * 4, hidden_dim * 2)

    def forward(self, token_seq, hidden_state, enc_output, hidden_type='cat'):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [1, batch_size, hidden_dim * 2]

            # 确保 hidden_state 是 3D 的
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        dec_output, hidden = self.rnn(embedded, hidden_state)

        # attention运算
        c_t = self.atteniton(enc_output, dec_output)
        # [attention, dec_output]
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        # 线性运算
        out = torch.tanh(self.atteniton_fc(cat_output))

        # out: [batch_size, seq_len, hidden_dim * 2]
        logits = self.fc(out)
        return logits, hidden

class Seq2Seq(nn.Module):

    def __init__(self,
                 enc_emb_size, 
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5,
                 hidden_type='cat' #添加hidden_type参数
                 ):
        
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout=dropout)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout=dropout)
        self.hidden_type = hidden_type  # 存储 hidden_type
        self.hidden_transform = nn.Linear(hidden_size, hidden_size * 2)



    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state, outputs = self.encoder(enc_input, self.hidden_type)

        if self.hidden_type == 'sum' or self.hidden_type == 'mul' :
            encoder_state = self.hidden_transform(encoder_state)
        output,hidden = self.decoder(dec_input, encoder_state, outputs, self.hidden_type)

        return output,hidden




#测试
import torch
import pickle


# 加载训练好的模型和词典
state_dict = torch.load('/Users/peiqi/code/AiPremiumClass/李思佳/week08/seq2seq_mul_state.bin')
with open('/Users/peiqi/code/AiPremiumClass/李思佳/week08/vocab.bin','rb') as f:
    evoc,dvoc = pickle.load(f)

model = Seq2Seq(
    enc_emb_size=len(evoc),
    dec_emb_size=len(dvoc),
    emb_dim=100,
    hidden_size=120,
    dropout=0.5,
    hidden_type='mul'
)
model.load_state_dict(state_dict)

# 创建解码器反向字典
dvoc_inv = {v:k for k,v in dvoc.items()}


# 用户输入
enc_input = "珍 藏 惟 有 诗 三 卷"
enc_idx = torch.tensor([[evoc[tk] for tk in enc_input.split()]])
print(enc_idx.shape)

# 推理
# 最大解码长度
max_dec_len = 50

model.eval()
with torch.no_grad():
    # 编码器
    # hidden_state = model.encoder(enc_idx)
    hidden_state, enc_outputs = model.encoder(enc_idx, hidden_type='cat')  # attention

    # 解码器输入 shape [1,1]
    dec_input = torch.tensor([[dvoc['BOS']]])

    # 循环decoder
    dec_tokens = []
    while True:
        if len(dec_tokens) >= max_dec_len:
            break
        # 解码器 
        # logits: [1,1,dec_voc_size]
        # logits,hidden_state = model.decoder(dec_input, hidden_state)
        logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs, hidden_type='cat')
            # 确保 hidden_state 是 3D 的
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
        
        # 下个token index
        next_token = torch.argmax(logits, dim=-1)

        if dvoc_inv[next_token.squeeze().item()] == 'EOS':
            break
        # 收集每次token_index 【解码集合】
        dec_tokens.append(next_token.squeeze().item())
        # decoder的下一个输入 = token_index
        dec_input = next_token
        hidden_state = hidden_state.view(1, -1)

# 输出解码结果
print(''.join([dvoc_inv[tk] for tk in dec_tokens]))

