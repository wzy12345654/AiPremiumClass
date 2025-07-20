import torch
import torch.nn as nn
from config import config
import numpy as np

class Encoder(nn.Module):
    """编码器模块，使用双向LSTM"""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.rnn = nn.LSTM(config['embed_dim'], 
                          config['enc_hid_dim'],
                          num_layers=config['n_layers'],
                          bidirectional=config['bidirectional'],
                          dropout=config['dropout'] if config['n_layers']>1 else 0)
        
        # 合并双向隐藏状态的线性层
        self.fc = nn.Linear(config['enc_hid_dim']*2 if config['bidirectional'] else config['enc_hid_dim'], 
                          config['dec_hid_dim'])

    def forward(self, src, src_len):
        # 词嵌入
        embedded = self.embedding(src)
        
        # 打包序列以提高效率
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)  # 解包
        
        # 处理双向LSTM的隐藏状态
        if config['bidirectional']:
            hidden = self.merge_hidden(hidden)
            cell = self.merge_hidden(cell)
        
        return outputs, (hidden, cell)

    def merge_hidden(self, hidden):
        """合并双向LSTM的隐藏状态"""
        hidden = hidden.view(config['n_layers'], 2, -1, config['enc_hid_dim']) 
        hidden = hidden[-1]
        if config['merge_mode'] == 'concat':  # 合并模式
            hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        else:
            hidden = hidden[0] + hidden[1]
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)
        return hidden

class Attention(nn.Module):
    """注意力机制模块"""
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(config['enc_hid_dim']*2 + config['dec_hid_dim'], config['dec_hid_dim'])
        self.v = nn.Linear(config['dec_hid_dim'], 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # 扩展隐藏状态维度以匹配编码器输出
        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        
        # 计算注意力权重
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=0)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.attention = Attention()
        self.rnn = nn.LSTM(config['embed_dim'] + config['enc_hid_dim']*2,  
                         config['dec_hid_dim'],  
                         num_layers=config['n_layers'],  
                         dropout=config['dropout'] if config['n_layers']>1 else 0)  
        self.fc = nn.Linear(config['dec_hid_dim'] + config['embed_dim'] + config['enc_hid_dim']*2, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        
        # 计算上下文向量
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.permute(1,0).unsqueeze(1),
                          encoder_outputs.permute(1,0,2)).permute(1,0,2)
        
        # 拼接输入和上下文
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # 生成最终预测
        output = torch.cat((output.squeeze(0),
                          embedded.squeeze(0),
                          context.squeeze(0)), dim=1)
        prediction = self.fc(output)
        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    """完整的Seq2Seq模型"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_len, trg):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        
        # 初始化输出张量
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(config['device'])  
        
        # 编码器前向传播
        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)
        
        # 初始输入为<sos>
        input = trg[0,:]
        
        # 逐步解码
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            # 教师强制策略
            teacher_force = np.random.random() < config['teacher_forcing_ratio']  
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs