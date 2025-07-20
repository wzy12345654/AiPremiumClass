import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import pickle


class Config:
    upper_file_path = './in.txt'
    lower_file_path = './out.txt'
    batch_size = 256
    emb_dim = 100
    hidden_dim = 120
    dropout = 0.5


class Vocabulary:
    def __init__(self):
        self.vocab = set()
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}

    def get_vocabulary(self, upper_couplets, lower_couplets):
        for upper in upper_couplets:
            self.vocab.update(upper)
        for lower in lower_couplets:
            self.vocab.update(lower)

    def get_two_dicts(self):
        self.word2idx.update({char: idx + 4 for idx, char in enumerate(self.vocab)})
        self.idx2word.update({idx + 4: char for idx, char in enumerate(self.vocab)})


class DataProcessor:
    def __init__(self):
        self.upper_couplets = []
        self.lower_couplets = []
        self.encoder_sequence = []
        self.decoder_input_sequence = []
        self.decoder_output_sequence = []
        self.data_set = None
        pass

    @staticmethod
    def load_data(data_path):
        with open(data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        return [
            [char for char in line.strip() if char != ' ']  # 过滤空格
            for line in lines
        ]

    def load_couplets_data(self):
        self.upper_couplets = self.load_data(Config.upper_file_path)
        self.lower_couplets = self.load_data(Config.lower_file_path)

    def get_sequence(self, vocab: Vocabulary):
        self.encoder_sequence = [[vocab.word2idx.get(word, 0) for word in scentence] for scentence in self.upper_couplets]
        self.decoder_input_sequence = [[vocab.word2idx['<bos>']]+[vocab.word2idx.get(word, 0) for word in scentence] for scentence in self.lower_couplets]
        self.decoder_output_sequence = [[vocab.word2idx.get(word, 0) for word in scentence] + [vocab.word2idx['<eos>']] for scentence in self.lower_couplets]
        self.data_set = [
            (torch.tensor(u), torch.tensor(l),torch.tensor(p))
            for u, l, p in zip(self.encoder_sequence, self.decoder_input_sequence,self.decoder_output_sequence)
        ]


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout):
        super(Encoder, self).__init__()
        # 定义嵌入层
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # 定义GRU层
        self.rnn = nn.GRU(emb_dim, hidden_dim, dropout=dropout,
                          batch_first=True, bidirectional=True)

    def forward(self, token_seq):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        outputs, hidden = self.rnn(embedded)
        # 返回，Encoder最后一个时间步的隐藏状态(拼接)
        # return outputs[:, -1, :]
        # 返回最后一个时间步的隐藏状态(拼接), 所有时间步的输出（attention准备）
        return torch.cat((hidden[0], hidden[1]), dim=1), outputs
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

    def forward(self, token_seq, hidden_state, enc_output):
        # token_seq: [batch_size, seq_len]
        # embedded: [batch_size, seq_len, emb_dim]
        embedded = self.embedding(token_seq)
        # outputs: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [1, batch_size, hidden_dim * 2]
        dec_output, hidden = self.rnn(embedded, hidden_state.unsqueeze(0))

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
                 ):
        super().__init__()

        # encoder
        self.encoder = Encoder(enc_emb_size, emb_dim, hidden_size, dropout=dropout)
        # decoder
        self.decoder = Decoder(dec_emb_size, emb_dim, hidden_size, dropout=dropout)

    def forward(self, enc_input, dec_input):
        # encoder last hidden state
        encoder_state, outputs = self.encoder(enc_input)
        output, hidden = self.decoder(dec_input, encoder_state, outputs)

        return output, hidden

def collate_fn(batch):
    # 获取当前 batch 中的上联和下联
    encoder_inputs, decoder_inputs, decoder_outputs = zip(*batch)

    # 对上联和下联进行填充，使它们的长度与当前 batch 中的最大长度相同
    encoder_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=1)  # padding_value为<pad>的索引
    decoder_input_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=1)
    decoder_output_padded = pad_sequence(decoder_outputs, batch_first=True, padding_value=1)

    return encoder_padded, decoder_input_padded, decoder_output_padded


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for enc_input, dec_input, dec_output in tqdm(data_loader, desc="Training"):
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        dec_output = dec_output.to(device)

        optimizer.zero_grad()
        # 模型输出：[batch_size, seq_len, vocab_size]
        output, _ = model(enc_input, dec_input)

        # 将输出变为 [batch_size * seq_len, vocab_size]
        output = output.view(-1, output.shape[-1])
        # 将目标变为 [batch_size * seq_len]
        dec_output = dec_output.view(-1)

        loss = criterion(output, dec_output)
        loss.backward()
        optimizer.step()
    return loss.item()


def main():
    device = torch.device("cuda")

    data_processor = DataProcessor()
    data_processor.load_couplets_data()

    vocabulary = Vocabulary()
    vocabulary.get_vocabulary(data_processor.upper_couplets, data_processor.lower_couplets)
    vocabulary.get_two_dicts()

    # 保存词汇表
    with open('./vocab.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print("词汇表已保存！")

    data_processor.get_sequence(vocabulary)
    data_loader = DataLoader(data_processor.data_set, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)

    input_dim = len(vocabulary.word2idx)
    output_dim = len(vocabulary.word2idx)

    model = Seq2Seq(
        enc_emb_size=input_dim,
        dec_emb_size=output_dim,
        emb_dim=Config.emb_dim,
        hidden_size=Config.hidden_dim,
        dropout=Config.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<pad>

    num_epochs = 20
    for epoch in range(num_epochs):
        loss = train(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "seq2seq_model.pt")
    print("模型训练完成并保存！")


if __name__ == '__main__':
    main()
