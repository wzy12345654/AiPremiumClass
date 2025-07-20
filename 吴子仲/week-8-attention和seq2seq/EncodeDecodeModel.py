import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout,
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        # 嵌入
        embedded = self.embedding(token_seq)
        # GRU前向传播
        outputs, hidden = self.rnn(embedded)
        # return hidden.squeeze(0)
        # 双向GRU的隐藏状态拼接
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        return hidden
        # 双向GRU的隐藏状态相加
        # return hidden.sum(dim=0)

class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=dropout,
                          batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, input_dim) # 解码词典中词汇概率分布

    def forward(self, token_seq, hidden_state):
        # 嵌入
        embedded = self.embedding(token_seq)
        # GRU前向传播
        # hidden_state: (batch_size, hidden_dim * 2) -> (1, batch_size, hidden_dim * 2)
        outputs, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))
        # 全连接层输出
        output = self.fc(outputs)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc_input_dim, dec_input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.encoder = Encoder(enc_input_dim, emb_dim, hidden_dim, dropout)
        self.decoder = Decoder(dec_input_dim, emb_dim, hidden_dim, dropout)
        
    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state = self.encoder(enc_input)
        # decoder output
        dec_output, hidden = self.decoder(dec_input, encoder_state)
        return dec_output, hidden

if __name__ == "__main__":

    # 定义超参数
    input_dim  = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10

    # # 创建Encoder实例
    # encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
    # # 创建随机输入数据
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # # 前向传播
    # hidden_state = encoder(token_seq)
    # print(hidden_state.shape)

    # # 测试Decoder
    # decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # logits = decoder(token_seq, hidden_state)

    # 构建Seq2Seq模型
    seq2seq = Seq2Seq(input_dim, input_dim, emb_dim, hidden_dim, dropout)

    # 将随机输入数据带入前向运算
    logits, hidden = seq2seq(
        enc_input = torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input = torch.randint(0, input_dim, (batch_size, seq_len))
    )

    print(logits.shape)