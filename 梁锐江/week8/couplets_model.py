import torch
import torch.nn as nn


class CoupletsEncodeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        embedded = self.embedding(token_seq)
        enc_outputs, hidden = self.rnn(embedded)
        # hidden shape [2,batch_size,hidden_dim]
        states = hidden.split(1, dim=0)
        return enc_outputs, torch.cat((states[0], states[1]), dim=-1)


class CoupletsDecodeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim * 2, batch_first=True)

        self.attention = CoupletsAttentionModel()
        self.attention_fc = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.linear = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, decode_seq, hidden, enc_outputs):
        embedded = self.embedding(decode_seq)
        # hidden shape [1,batch_size,hidden_dim * 2]
        # dec_outputs shape [batch_size, seq_len, hidden_dim * 2]
        dec_outputs, dec_hidden = self.rnn(embedded, hidden)
        c_t = self.attention(enc_outputs, dec_outputs)
        scores = torch.tanh(self.attention_fc(torch.cat((c_t, dec_outputs), dim=-1)))
        logits = self.linear(scores)

        return logits, hidden


class CoupletsAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, enc_outputs, dec_outputs):
        # enc_outputs shape [batch_size, src_len, hidden_dim * 2]
        # dec_outputs shape [batch_size, tgt_len, hidden_dim * 2]
        s_t = torch.bmm(dec_outputs, enc_outputs.transpose(1, 2))
        # s_t shape [batch_size, src_len, tgt_len]
        a_t = torch.softmax(s_t, dim=-1)

        # a_t.transpose(1,2) shape [batch_size, tgt_len, src_len]
        c_t = torch.bmm(a_t, enc_outputs)
        # c_t shape [batch_size, tgt_len, hidden_dim * 2]
        return c_t


class CoupletsSeq2SeqModel(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = CoupletsEncodeModel(enc_vocab_size, embedding_dim, hidden_dim)
        self.decoder = CoupletsDecodeModel(dec_vocab_size, embedding_dim, hidden_dim)

    def forward(self, in_inputs, out_inputs):
        enc_outputs, enc_hidden = self.encoder(in_inputs)
        logits, hidden = self.decoder(out_inputs, enc_hidden, enc_outputs)
        return logits, hidden


if __name__ == '__main__':
    enc_vocab_size = 100
    dec_vocab_size = 105
    embedding_dim = 120
    hidden_dim = 110
    batch_size = 5000
    seq_len = 20

    # 测试模型是否能跑
    enc_inputs = torch.randint(0, enc_vocab_size, (batch_size, seq_len))
    dec_inputs = torch.randint(0, dec_vocab_size, (batch_size, seq_len))

    model = CoupletsSeq2SeqModel(enc_vocab_size, dec_vocab_size, embedding_dim, hidden_dim)
    logits, _ = model.forward(enc_inputs, dec_inputs)
    print(logits.shape)
