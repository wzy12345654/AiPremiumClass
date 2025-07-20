import torch
import torch.nn as nn


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim,dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        # 返回最后一个时间步的隐藏状态(拼接)
        return torch.cat((hidden[0], hidden[1]), dim=1)
        # 返回最后一个时间步的隐状态（相加）
        # return hidden.sum(dim=0)
# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Decoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim * 2, dropout=dropout, batch_first=True)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # 解码词典中词汇概率

    def forward(self, token_seq, hidden_state):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [1, batch_size, hidden_dim * 2]
        outputs, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))
        # logits: [batch_size, seq_len, input_dim]
        logits = self.fc(outputs)
        return logits, hidden

class Seq2Seq(nn.Module):

    def __init__(self,
                 enc_emb_size, 
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5):
        
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout=dropout)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout=dropout)


    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state = self.encoder(enc_input)
        output,hidden = self.decoder(dec_input, encoder_state)

        return output,hidden
    

    
if __name__ == '__main__':
    
    # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10

    # encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # hidden_state = encoder(token_seq)  # Encoder输出（最后时间步状态）
    # # print(hidden_state.shape)  # 应该是 [batch_size, hidden_dim]

    # # 测试Decoder
    # decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # logits = decoder(token_seq, hidden_state)  # Decoder输出
    # print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )

    logits = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]