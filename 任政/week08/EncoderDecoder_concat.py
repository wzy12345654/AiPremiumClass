import torch
import torch.nn as nn

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义双向GRU
        self.gru = nn.GRU(emb_dim, hidden_dim, bidirectional=True, dropout=dropout  ,batch_first = True)
    # 前向传播
    def forward(self, token_seq):
        embedded = self.embedding(token_seq)
        outputs , hidden = self.gru(embedded)
        # 返回最后一个时间步的隐藏状态（拼接方式）
        # 因为是双向GRU，所以需要将两个方向的隐藏状态拼接起来
        # hidden[0] 是正向GRU的隐藏状态
        # hidden[1] 是反向GRU的隐藏状态
        # hidden[0].shape = [batch_size, seq_len, hidden_dim]
        # hidden[1].shape = [batch_size, seq_len, hidden_dim]
        # 所以我们需要将两个方向的隐藏状态拼接起来
        # 拼接方式是将两个方向的隐藏状态的最后一个时间步拼接起来
        # 所以我们需要将 hidden[0][-1] 和 hidden[1][-1] 拼接起来
        # 拼接后形状为 [batch_size, hidden_dim * 2]
        # concat 为拼接 所以维度会翻倍 hidden_dim * 2
        return torch.cat((hidden[0] , hidden[1]), dim=1) , outputs
        # add 相加  合并信息  保持原始维度hidden_dim

# 添加Attention
class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , enc_output , dec_output):
        # 计算权重注意力
        # 编码器输出 与 解码器输出 进行点积运算
        # permute 是 将 解码器输出的矩阵进行位置交换，第三个位置换到第二个位置
        # 因为 解码器输出的形状是 [batch_size, seq_len, hidden_dim]
        # 而点积运算需要两个矩阵的最后一个维度相同，所以需要将解码器输出的形状转换为 [batch_size, hidden_dim, seq_len]
        # 这样才能进行点积运算
        a_t = torch.bmm(enc_output , dec_output.permute(0, 2, 1))
        # 对权重进行softmax归一化
        # 因为我们需要计算每个时间步的权重，所以需要对最后一个维度进行softmax归一化
        a_t = torch.softmax(a_t , dim = -1)
        # 对编码器输出进行加权
        c_t = torch.bmm(a_t.permute(0 , 2 , 1) , enc_output)
        # 返回加权后的编码器输出
        return c_t

# 解码器
class Decoder(nn.Module):
    def __init__(self , input_dim , emb_dim , hidden_dim , dropout):
        super(Decoder , self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim , emb_dim)
        # 定义 GRU hidden_dim * 2 是为了跟编码器的双向GRU保持一致
        self.rnn = nn.GRU(emb_dim , hidden_dim * 2 , dropout = dropout , batch_first = True)
        # 定义线性层
        self.fc = nn.Linear(hidden_dim * 2, input_dim)  # 输入维度为隐藏层维度的两倍
        # 定义注意力层
        self.attention = Attention()
        # 将注意力层转换为线性层
        self.attention_fc = nn.Linear(hidden_dim * 4 , hidden_dim *2)
    # 前向传播
    def forward(self , token_seq , hidden_state , enc_ouput):
        # 嵌入层

        embedded = self.embedding(token_seq)
        # 将2D扩展到3D hidden_state 的形状从 [batch_size, hidden_dim * 2] 变为 [1, batch_size, hidden_dim * 2]，符合 GRU 的输入要求
        hidden_state = hidden_state.unsqueeze(0)
        dec_outputs , hidden = self.rnn(embedded, hidden_state)

        # 添加attention运算
        c_t  = self.attention(enc_ouput , dec_outputs)
        # 最后一个维度进行拼接注意力层的输出和解码器的输出
        cat_output = torch.cat((c_t , dec_outputs), dim = -1)
        # 先进行线性运算，然后tanh转换成非线性
        cat_output = torch.tanh(self.attention_fc(cat_output))
        logis = self.fc(cat_output)
        return logis, hidden

# 序列到序列模型
class Seq2Seq(nn.Module):
    def __init__(self, enc_emb_size, dec_emb_size, emb_dim, hidden_size, dropout = 0.5):
        super(Seq2Seq, self).__init__()
        # 编码器
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout)
        # 解码器
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout)

    def forward(self , enc_input , dec_input):
        # 编码器的隐藏状态
        encoder_state , outputs = self.encoder(enc_input)
        # 解码器的输出
        output , hidden = self.decoder(dec_input, encoder_state , outputs)
        return output , hidden

if __name__ == '__main__':
    # 测试Encoder
    input_dim = 200
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 4
    seq_len = 10
    # 测试Encoder
    encoder = Encoder(input_dim, emb_dim, hidden_dim, dropout)
    token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    hidden_state = encoder(token_seq)  # Encoder输出（最后时间步状态）
    print(hidden_state.shape)  # 应该是 [batch_size, hidden_dim]
    # 测试Decoder
    decoder = Decoder(input_dim, emb_dim, hidden_dim, dropout)
    token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    logis = decoder(token_seq, hidden_state)  # Decoder输出
    print(logis.shape)  # 应该是 [batch_size, seq_len, input_dim]
    # 测试Seq2Seq
    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )
    logis = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logis.shape)  # 应该是 [batch_size, seq_len, input_dim]























