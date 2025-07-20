import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, dropout, hidden_mode):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout if hidden_mode['num_layers'] > 1 else 0)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers=hidden_mode['num_layers'],
            dropout=dropout if hidden_mode['num_layers'] > 1 else 0,
            bidirectional=hidden_mode['bidirectional'])
        self.hidden_mode = hidden_mode
        self.fc = nn.Linear(2 * hidden_size, hidden_size) if hidden_mode == 'concat' else None

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)

        # 处理双向隐藏状态
        if self.hidden_mode == 'concat':
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch, 2*hidden]
        elif self.hidden_mode == 'add':
            hidden = hidden[-2] + hidden[-1]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [num_layers, batch_size, dec_hid_dim]
        encoder_outputs: [src_len, batch_size, enc_hid_dim * 2]
        """
        # 1. 取最后一层的隐藏状态 [batch_size, dec_hid_dim]
        decoder_hidden = decoder_hidden[-1]  # → [128, 512]

        # 2. 调整 encoder_outputs 至 [batch_size, src_len, enc_hid_dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [128, 25, 1024]

        # 3. 广播 decoder_hidden 至 [batch_size, src_len, dec_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)  # [128, 25, 512]

        # 4. 拼接至 [batch, src_len, dec_hid_dim + enc_hid_dim * 2]
        concat = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # [128, 25, 1536]

        # 5. 验证 self.attn 的输入维度是否匹配
        print(f"concat.shape = {concat.shape}")  # 必须等于 [128, 25, 1536]
        print(f"self.attn = {self.attn}")  # 必须为 Linear(in_features=1536, out_features=dec_hid_dim)

        # 6. 计算注意力得分
        energy = torch.tanh(self.attn(concat))  # [128, 25, dec_hid_dim]
        attention = self.v(energy).squeeze(2)  # [128, 25]
        return F.softmax(attention, dim=1)  # [128, 25]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, num_layers):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim, num_layers=num_layers)
        self.fc = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, decoder_hidden, encoder_outputs):
        # 词嵌入
        embedded = self.dropout(self.embedding(dec_input)).unsqueeze(0)  # [1, batch, emb_dim]

        # 注意力计算
        attn_weights = self.attention(decoder_hidden, encoder_outputs)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]

        # 3. 计算上下文向量 (加权编码器输出)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid*2]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, enc_hid*2]
        context = context.transpose(0, 1)  # [1, batch, enc_hid*2] (对齐序列维度)

        # 4. 准备RNN输入：拼接词嵌入和上下文
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch, emb_dim+enc_hid*2]

        # 5. GRU处理
        output, hidden = self.rnn(rnn_input, decoder_hidden)  # output: [1, batch, dec_hid]

        # 6. 最终预测
        embedded = embedded.squeeze(0)  # [batch, emb_dim]
        output = output.squeeze(0)  # [batch, dec_hid]
        context = context.squeeze(0)  # [batch, enc_hid*2]
        prediction = self.fc(torch.cat((output, context, embedded), dim=1))  # [batch, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, config, enc_vocab, dec_vocab):
        super().__init__()
        self.config = config
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

        # 新增双向维度计算
        self.enc_hid_dim = config.hidden_size
        self.dec_hid_dim = config.hidden_size
        self.num_directions = 2 if config.enc_hidden_mode['bidirectional'] else 1

        # 构建注意力模块
        self.attention = Attention(
            self.enc_hid_dim * self.num_directions,
            self.dec_hid_dim
        )

        # 修改后的Encoder/Decoder初始化
        self.encoder = Encoder(
            len(enc_vocab),
            config.emb_dim,
            self.enc_hid_dim,
            config.dropout,
            config.enc_hidden_mode
        )
        self.decoder = Decoder(
            output_dim=len(dec_vocab),
            emb_dim=config.emb_dim,
            enc_hid_dim=self.enc_hid_dim * self.num_directions,
            dec_hid_dim=self.dec_hid_dim,
            dropout=config.dropout,
            attention=self.attention,  # 传递注意力模块
            num_layers=config.enc_hidden_mode['num_layers']
        )

        # 若encoder双向则需调整传递给decoder的初始hidden
        if self.num_directions == 2:
            self.reduce_hidden = nn.Linear(
                self.enc_hid_dim * self.num_directions,
                self.dec_hid_dim
            )

    def _init_decoder_hidden(self, encoder_hidden):
        """处理双向Encoder的隐状态"""
        if self.num_directions == 1:
            return encoder_hidden

        # 双向GRU：合并每层的正向和反向最后隐状态
        # encoder_hidden形状 [num_lay*2, batch, enc_hid]
        encoder_hidden = encoder_hidden.view(
            self.config.enc_hidden_mode['num_layers'],
            2,  # num_directions
            -1,  # batch_size
            self.enc_hid_dim
        )

        # 取各层最后一个时间步的隐状态合并（cat方式）
        last_layer_hidden = torch.cat([
            encoder_hidden[-1, 0, :, :],  # last layer forward
            encoder_hidden[-1, 1, :, :]  # last layer backward
        ], dim=1)

        transformed_hidden = self.reduce_hidden(last_layer_hidden)

        return transformed_hidden.unsqueeze(0).repeat(self.config.enc_hidden_mode['num_layers'], 1, 1)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """
        调整后的维度流程：
        src → [batch, src_len]
        trg → [batch, tgt_len] (训练时可空)
        """
        batch_size = src.size(0)
        max_len = trg.size(1) if trg is not None else self.config.max_length

        # 转置src满足Encoder的时序维度要求 [src_len, batch]
        src = src.permute(1, 0)

        # Encoder前向 → encoder_outputs [src_len, batch, enc_hid*2]
        #               hidden [num_lay*2, batch, enc_hid]
        encoder_outputs, encoder_hidden = self.encoder(src)

        # 转换初始Decoder隐状态 [num_lay, batch, dec_hid]
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)

        # 准备Decoder初始输入 → [batch,]
        dec_inputs = torch.full(
            (batch_size,),
            self.dec_vocab.bos_idx,
            device=src.device
        )

        outputs = []
        for t in range(max_len):
            # decoder_output → [batch, output_dim]
            # decoder_hidden → [num_lay, batch, dec_hid]
            decoder_output, decoder_hidden = self.decoder(
                dec_input=dec_inputs,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs
            )
            outputs.append(decoder_output)

            # 决定下一步的输入（注意处理时序）
            teacher_force = (trg is not None) and (random.random() < teacher_forcing_ratio)
            if teacher_force:
                dec_inputs = trg[:, t]
            else:
                dec_inputs = decoder_output.argmax(1)

        # 调整输出为三维张量 [batch, tgt_len, output_dim]
        return torch.stack(outputs, dim=1)
