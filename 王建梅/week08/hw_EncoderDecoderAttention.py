import torch
import torch.nn as nn
import torch.nn.functional as F

# 加入注意力机制的编码器和解码器，使用GRU，实现一个seq2seq模型
class Encoder(nn.Module):
    def __init__(self, input_dim,embedding_dim, hidden_dim,dropout,cat_or_add='cat'):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.cat_or_add = cat_or_add
        # 定义嵌入层,input_dim 是词典大小,embedding_dim 是词嵌入向量维度
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # 定义双向GRU层
        self.rnn = nn.GRU(embedding_dim, hidden_dim,dropout=dropout,
                          batch_first=True,bidirectional=True)

    def forward(self,token_seq):
        # 输入token序列，输出隐藏状态
        # token_seq: [batch_size, seq_len] 输入的是一个batch的token序列，每个token是一个词典中的索引值
        # embedded: [batch_size, seq_len, embedding_dim]
        # 从Embedding模块的权重矩阵中，取出对应索引的词向量，得到每个token的词向量表示
        embedded = self.embedding(token_seq)
        # 所有时间步的输出和最后一个时间步的隐藏状态
        # outputs: [batch_size, seq_len, hidden_dim*2]  为什么是这个形状？因为有两个方向，每个方向的隐藏状态是hidden_dim
        # hidden: [2, batch_size, hidden_dim]  outputs的最后一个时间步的隐藏状态，是两个方向的隐藏状态拼接起来的吗？
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        # 返回最后一个时间步的隐藏状态(拼接/相加)
        # allsteps = torch.cat((hidden[0], hidden[1]), dim=1)
        if self.cat_or_add == 'add':
            last_hidden_state = torch.add(hidden[0], hidden[1]) # [batch_size, hidden_dim]
            # outputs维度降低，正向反向隐藏状态相加
            outputs = outputs[:,:,0:self.hidden_dim] + outputs[:,:,self.hidden_dim:] # [batch_size, seq_len, hidden_dim]
        else:
            # 拼接两个方向的隐藏状态
            last_hidden_state = torch.cat((hidden[0], hidden[1]), dim=1) # [batch_size, hidden_dim*2]
        return outputs,last_hidden_state

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, enc_output, dec_output):
        # enc_output: [batch_size, seq_len_enc, hidden_dim]
        # dec_output: [batch_size, seq_len_dec, hidden_dim]
        # 计算注意力权重
        # a_t = h_t @ h_s  
        # a_t：[batch_size, seq_len_enc, seq_len_dec]，
        # 其中每个元素 a_t[i, j, k] 表示第 i 个样本中，编码器输出序列的第 j 个时间步与解码器输出序列的第 k 个时间步之间的注意力分数
        a_t = torch.bmm(enc_output, dec_output.permute(0, 2, 1))
        # 1.计算 结合解码token和编码token，关联的权重
        a_t = torch.softmax(a_t, dim=1)
        # 2.计算 关联权重和编码token 贡献值
        # c_t: [batch_size, seq_len_dec, hidden_dim]
        c_t = torch.bmm(a_t.permute(0, 2, 1), enc_output)
        return c_t

class Decoder(nn.Module):
    def __init__(self, output_dim,embedding_dim, hidden_dim,dropout,cat_or_add='cat'):
        super(Decoder, self).__init__()
        # 定义嵌入层，output_dim 是词典大小,embedding_dim 是词嵌入向量维度
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        if cat_or_add == 'add':
            dec_hidden_dim = hidden_dim
        else:
            dec_hidden_dim = hidden_dim*2
        # 定义GRU层
        self.rnn = nn.GRU(embedding_dim, dec_hidden_dim,dropout=dropout,
                          batch_first=True)
        # 定义线性层，解码词典中词汇概率
        self.fc = nn.Linear(dec_hidden_dim, output_dim) 
        # 定义注意力层
        self.attention = Attention()
        # 定义注意力输出层
        self.attention_fc = nn.Linear(dec_hidden_dim * 2, dec_hidden_dim)

    def forward(self, token_seq, enc_hidden_state, enc_output):
        # 输入token序列，输出隐藏状态和输出
        # token_seq: [batch_size, 1]
        # hidden_state: [batch_size, dec_hidden_dim]
        # enc_output: [batch_size, seq_len, dec_hidden_dim]
        # embedded: [batch_size, 1, embedding_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, dec_hidden_dim] dec_hidden_dim = hidden_dim*2 or hidden_dim 取决于cat_or_add
        # hidden: [1, batch_size, dec_hidden_dim]
        dec_output, hidden = self.rnn(embedded, enc_hidden_state.unsqueeze(0))
        # attention运算
        # c_t: [batch_size, 1, dec_hidden_dim] 
        # 计算注意力权重，得到上下文向量，c_t 的形状与 dec_output 相同，
        # 等同于将dec_output中的每个token与enc_output中的每个token进行加权求和，得到dec_output中每个token的上下文向量
        c_t = self.attention(enc_output, dec_output)
        # attention 输出层
        cat_output = torch.cat((c_t,dec_output), dim=-1) # [batch_size, dec_hidden_dim*2]
        out = torch.tanh(self.attention_fc(cat_output)) # [batch_size, dec_hidden_dim]
        # 计算输出概率
        # output: [batch_size, output_dim]
        output = self.fc(out) 
        # 返回输出和隐藏状态
        return output, hidden

# 定义seq2seq模型
class Seq2Seq(nn.Module):
    def __init__(self, enc_emb_size, dec_emb_size,emb_dim,hidden_size,dropout=0.5,cat_or_add='cat'):
        super(Seq2Seq, self).__init__()
        # 定义编码器和解码器
        self.encoder = Encoder(enc_emb_size,emb_dim, hidden_size,dropout,cat_or_add)
        self.decoder = Decoder(dec_emb_size,emb_dim, hidden_size,dropout,cat_or_add)
        
    def forward(self, enc_input, dec_input):
        # 输入编码器和解码器的输入序列，输出解码器的输出序列
        # enc_input: [batch_size, seq_len]
        # dec_input: [batch_size, seq_len]
        # enc_output: [batch_size, seq_len, hidden_dim*2]
        # encoder last hidden state
        enc_output, enc_hidden = self.encoder(enc_input)
        # output: [batch_size, seq_len, output_dim]
        # hidden: [1, batch_size, dec_hidden_dim]
        output, hidden = self.decoder(dec_input, enc_hidden, enc_output)
        return output, hidden

if __name__ == '__main__':
    # 定义参数
    input_dim = 10   # encode词典大小
    output_dim = 10  # 输出词典大小
    embedding_dim = 4  # 词嵌入向量维度
    hidden_dim = 8  # rnn隐藏层维度
    dropout = 0.5
    batch_size = 2
    seq_len = 5  # 序列长度
    cat_or_add='cat' # 编码器隐藏层拼接/相加

    # 测试Encoder
    # encoder = Encoder(input_dim, embedding_dim, hidden_dim, dropout,cat_or_add)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # hidden_state_cat,enc_output_cat = encoder(token_seq)  # Encoder输出（最后时间步状态）
    # hidden_state_add,enc_output_add = encoder(token_seq)  # Encoder输出（最后时间步状态）
    # print(hidden_state_cat.shape,hidden_state_cat)  # [batch_size, hidden_dim*2]
    # print(hidden_state_add.shape,hidden_state_add)  # [batch_size, hidden_dim]

    # 测试Decoder
    # decoder = Decoder(input_dim, embedding_dim, hidden_dim, dropout,cat_or_add)
    # token_seq = torch.randint(0, input_dim, (batch_size, seq_len))
    # logits_cat = decoder(token_seq, hidden_state_cat,enc_output_cat)  # Decoder输出
    # logits_add = decoder(token_seq, hidden_state_add,enc_output_add)  # Decoder输出
    # print(logits_cat.shape,logits_cat)  # 应该是 [batch_size, seq_len, input_dim]
    # print(logits_add.shape,logits_add)  # 应该是 [batch_size, seq_len, input_dim] 

    # 测试Seq2Seq
    seq2seq_cat = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=output_dim,
        emb_dim=embedding_dim,
        hidden_size=hidden_dim,
        dropout=dropout
    )
    logits_cat,hidden_cat = seq2seq_cat(enc_input=torch.randint(0, input_dim, (batch_size, seq_len)),
                     dec_input=torch.randint(0, input_dim, (batch_size, seq_len)))
    print(logits_cat.shape)  # 应该是 [batch_size, seq_len, output_dim]
    print(hidden_cat.shape)  # 应该是 [batch_size, hidden_dim*2]

    seq2seq_add = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=output_dim,
        emb_dim=embedding_dim,
        hidden_size=hidden_dim,
        dropout=dropout,
        cat_or_add='add'
    )
    logits_add,hidden_add = seq2seq_add(enc_input=torch.randint(0, input_dim, (batch_size, seq_len))
                     ,dec_input=torch.randint(0, input_dim, (batch_size, seq_len)))
    print(logits_add.shape)  # 应该是 [batch_size, seq_len, output_dim]
    print(hidden_add.shape)  # 应该是 [batch_size, hidden_dim]
    #print(logits_cat)  # 应该是 [batch_size, seq_len, output_dim]
    #print(logits_add)  # 应该是 [batch_size, seq_len, output_dim]