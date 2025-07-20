import torch
import torch.nn as nn


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=2, state_type='concat'):
        """
        初始化编码器模块。

        Args:
            input_dim (int): 输入词汇表的大小，即不同 token 的数量。
            emb_dim (int): 嵌入层的维度，用于将输入的 token 转换为密集向量表示。
            hidden_dim (int): 循环神经网络（GRU）隐藏层的维度。
            num_layers (int, optional): GRU 层的数量，默认为 2。
            state_type (str, optional): 编码器最后一个时间步隐藏状态的处理方式，
                可选值为 'concat'（拼接） 或 'add'（相加），默认为 'concat'。
        """
        # 调用父类 nn.Module 的构造函数
        super(Encoder, self).__init__()
        # 定义嵌入层，将输入的 token 索引转换为对应的嵌入向量
        # input_dim 为输入词汇表的大小，emb_dim 为嵌入向量的维度
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义双向 GRU 层，用于处理序列数据
        # emb_dim 为输入特征维度，hidden_dim 为隐藏层维度
        # num_layers 为 GRU 层的堆叠数量，batch_first=True 表示输入输出的第一维为批次大小
        # bidirectional=True 表示使用双向 GRU
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        # 自定义返回的隐藏状态类型，可选择 'concat' 或 'add'
        self.state_type = state_type

    def forward(self, token_seq):
        """
        编码器的前向传播方法，对输入的 token 序列进行编码处理。

        Args:
            token_seq (torch.Tensor): 输入的 token 序列，形状为 [batch_size, seq_len]。

        Returns:
            tuple: 包含两个元素的元组，第一个元素是编码器最后一个时间步的隐藏状态，
                   第二个元素是编码器的所有时间步的输出。
        """
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        # 通过嵌入层将输入的 token 序列转换为对应的嵌入向量
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        # 经过双向 GRU 层处理嵌入向量，得到所有时间步的输出和最终的隐藏状态
        # 由于是双向 GRU，输出维度会翻倍
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接或相加)
        if self.state_type == 'concat':
            # 最后一个时间步的隐藏状态，将双向的隐藏状态进行拼接
            # 取所有批次的最后一个时间步的输出作为最终隐藏状态
            hidden = outputs[:,-1,:]
        elif self.state_type == 'add':
            # 返回最后一个时间步的隐藏状态(相加)
            # 将双向 GRU 的隐藏状态在第 0 维相加
            hidden = torch.sum(hidden, dim=0)
            # 将双向 GRU 的输出的前半部分和后半部分相加
            outputs = outputs[...,:250] + outputs[...,250:]
        else:
            # 若 state_type 不是 'concat' 或 'add'，抛出异常
            raise ValueError("state_type must be 'concat' or 'add'")
        return hidden, outputs


class Attention(nn.Module):
    def __init__(self):
        """
        初始化 Attention 类的实例。
        此方法会调用父类 nn.Module 的构造函数，完成必要的初始化操作。
        由于 Attention 模块目前不需要额外的可学习参数，构造函数内容较为简洁。
        """
        # 调用父类 nn.Module 的构造函数，确保父类的初始化逻辑被执行
        super().__init__()

    def forward(self, enc_output, dec_output):
        """
        计算注意力机制的上下文向量。

        Args:
            enc_output (torch.Tensor): 编码器的输出，形状为 [batch_size, enc_seq_len, hidden_dim]。
            dec_output (torch.Tensor): 解码器的输出，形状为 [batch_size, dec_seq_len, hidden_dim]。

        Returns:
            torch.Tensor: 上下文向量，形状为 [batch_size, dec_seq_len, hidden_dim]。
        """
        # a_t = h_t @ h_s  
        # 计算解码器输出与编码器输出之间的点积，得到注意力分数
        # enc_output 形状: [batch_size, enc_seq_len, hidden_dim]
        # dec_output.permute(0, 2, 1) 形状: [batch_size, hidden_dim, dec_seq_len]
        # a_t 形状: [batch_size, enc_seq_len, dec_seq_len]
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 1.计算 结合解码token和编码token，关联的权重
        # 对注意力分数应用 softmax 函数，将分数转换为概率分布
        # 沿着 enc_seq_len 维度进行 softmax 操作
        # a_t 形状: [batch_size, enc_seq_len, dec_seq_len]
        a_t = torch.softmax(a_t, dim=1)
        # 2.计算 关联权重和编码token 贡献值
        # 将注意力权重与编码器输出进行加权求和，得到上下文向量
        # a_t.permute(0, 2, 1) 形状: [batch_size, dec_seq_len, enc_seq_len]
        # enc_output 形状: [batch_size, enc_seq_len, hidden_dim]
        # c_t 形状: [batch_size, dec_seq_len, hidden_dim]
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t

# 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout=0.5, state_type='concat'):
        """
        初始化解码器模块。

        Args:
            input_dim (int): 输入词汇表的大小，即不同 token 的数量。
            emb_dim (int): 嵌入层的维度，用于将输入的 token 转换为密集向量表示。
            hidden_dim (int): 循环神经网络（GRU）隐藏层的维度。
            dropout (float, optional): Dropout 概率，用于防止过拟合，默认为 0.5。
            state_type (str, optional): 编码器最后一个时间步隐藏状态的处理方式，
                可选值为 'concat'（拼接） 或 'add'（相加），默认为 'concat'。
        """
        # 调用父类 nn.Module 的构造函数
        super(Decoder, self).__init__()
        # 如果编码器隐藏状态处理方式为 'concat'，则隐藏层维度翻倍
        if state_type == 'concat':
            hidden_dim = hidden_dim * 2
        # 定义嵌入层，将输入的 token 索引转换为对应的嵌入向量
        # input_dim 为输入词汇表的大小，emb_dim 为嵌入向量的维度
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义 GRU 层，用于处理序列数据
        # emb_dim 为输入特征维度，hidden_dim 为隐藏层维度
        # batch_first=True 表示输入输出的第一维为批次大小
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        # 定义线性层，将 GRU 层的输出映射到词汇表大小的维度
        # 用于计算解码词典中每个词汇的概率
        self.fc = nn.Linear(hidden_dim, input_dim)
        # 定义注意力层，用于计算上下文向量
        self.atteniton = Attention()
        # 定义线性层，将注意力层的输出和 GRU 层的输出拼接后的结果进行转换
        # 输入维度为隐藏层维度的两倍，输出维度为隐藏层维度
        self.atteniton_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        # 定义 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_seq, hidden_state, enc_output):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [1, batch_size, hidden_dim]
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))

        # attention运算
        c_t = self.atteniton(enc_output, dec_output)
        # [attention, dec_output]
        cat_output = torch.cat((c_t, dec_output), dim=-1)
        # 线性运算
        out = torch.tanh(self.atteniton_fc(cat_output))
        # dropout
        out = self.dropout(out)
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
                 state_type='concat'
                 ):
        """
        初始化 Seq2Seq 模型。

        Args:
            enc_emb_size (int): 编码器输入词汇表的大小，即编码器输入不同 token 的数量。
            dec_emb_size (int): 解码器输入词汇表的大小，即解码器输入不同 token 的数量。
            emb_dim (int): 嵌入层的维度，用于将输入的 token 转换为密集向量表示。
            hidden_size (int): 循环神经网络（GRU）隐藏层的维度。
            dropout (float, optional): Dropout 概率，用于防止过拟合，默认为 0.5。
            state_type (str, optional): 编码器最后一个时间步隐藏状态的处理方式，
                可选值为 'concat'（拼接） 或 'add'（相加），默认为 'concat'。
        """
        # 调用父类 nn.Module 的构造函数
        super().__init__()

        # 初始化编码器模块
        # enc_emb_size 为编码器输入词汇表大小
        # emb_dim 为嵌入层维度
        # hidden_size 为循环神经网络隐藏层维度
        # state_type 为编码器最后一个时间步隐藏状态的处理方式
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size,state_type=state_type)
        # 初始化解码器模块
        # dec_emb_size 为解码器输入词汇表大小
        # emb_dim 为嵌入层维度
        # hidden_size 为循环神经网络隐藏层维度
        # dropout 为 Dropout 概率
        # state_type 为编码器最后一个时间步隐藏状态的处理方式
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout,state_type=state_type)


    def forward(self, enc_input, dec_input):
        """
        Seq2Seq 模型的前向传播方法，将编码器输入和解码器输入进行处理，得到最终输出。

        Args:
            enc_input (torch.Tensor): 编码器的输入 token 序列，形状为 [batch_size, enc_seq_len]。
            dec_input (torch.Tensor): 解码器的输入 token 序列，形状为 [batch_size, dec_seq_len]。

        Returns:
            tuple: 包含两个元素的元组，第一个元素是解码器的输出 logits，形状为 [batch_size, dec_seq_len, input_dim]；
                   第二个元素是解码器的最终隐藏状态，形状为 [1, batch_size, hidden_dim]。
        """
        # 调用编码器的前向传播方法，对编码器输入进行编码处理
        # encoder_state 是编码器最后一个时间步的隐藏状态
        # outputs 是编码器所有时间步的输出
        encoder_state, outputs = self.encoder(enc_input)
        # 调用解码器的前向传播方法，结合编码器的最后隐藏状态和所有时间步输出，对解码器输入进行解码处理
        # output 是解码器的输出 logits，用于预测词汇表中每个词的概率
        # hidden 是解码器的最终隐藏状态
        output, hidden = self.decoder(dec_input, encoder_state, outputs)

        return output, hidden

if __name__ == '__main__':
    
    # 测试Encoder
    input_dim = 100
    emb_dim = 256
    hidden_dim = 256
    dropout = 0.5
    batch_size = 15
    seq_len = 10

    seq2seq = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=input_dim,
        emb_dim=emb_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )

    logits,_ = seq2seq(
        enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
        dec_input=torch.randint(0, input_dim, (batch_size, seq_len))
    )
    print(logits.shape)  # 应该是 [batch_size, seq_len, input_dim]

