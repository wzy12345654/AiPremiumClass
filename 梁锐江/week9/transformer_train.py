from torch.utils.data import Dataset, DataLoader
import torch
from exercise_transformer_model import Seq2SeqTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


def build_vocab(tokens):
    vocab = {'<pad>': 1, '<s>': 2, '</s>': 3}
    idx = 3
    for seq in tokens:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


class MyDataset(Dataset):

    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        super().__init__()
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def __len__(self):
        return len(self.enc_tokens)

    def __getitem__(self, index):
        enc_idx = [self.enc_vocab[token] for token in self.enc_tokens[index]]
        dec_idx = [self.dec_vocab[token] for token in self.dec_tokens[index]]
        return torch.tensor(enc_idx), torch.tensor(dec_idx)


def collate_fn(batch_data):
    enc_batch = [batch[0] for batch in batch_data]
    dec_batch = [batch[1][:-1] for batch in batch_data]
    dec_tgt_batch = [batch[1][1:] for batch in batch_data]

    enc_in = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    dec_in = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    dec_tgt = pad_sequence(dec_tgt_batch, batch_first=True, padding_value=0)
    return enc_in, dec_in, dec_tgt


if __name__ == '__main__':
    corpus = "人生得意须尽欢，莫使金樽空对月"
    chars = list(corpus)

    enc_tokens = []
    dec_tokens = []

    # writer = SummaryWriter()

    for i in range(1, len(chars)):
        enc = chars[:i]
        dec = ['<s>'] + chars[i:] + ['</s>']

        enc_tokens.append(enc)
        dec_tokens.append(dec)

    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)

    dataset = MyDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dl = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

    # 模型参数
    epochs = 500
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2SeqTransformer(d_model=d_model, enc_vocab_size=enc_voc_size, dec_vocab_size=dec_voc_size,
                               dropout=dropout)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # total_step = 0
    model.train()
    for epoch in range(epochs):
        for enc_in, dec_in, dec_tgt in dl:
            optimizer.zero_grad()
            enc_in, dec_in, dec_tgt = enc_in.to(device), dec_in.to(device), dec_tgt.to(device)

            outputs = model(enc_in, dec_in)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), dec_tgt.view(-1))

            loss.backward()
            optimizer.step()
            print(f'loss {loss.item()}')
        #     writer.add_scalar('loss', loss.item(), total_step)
        # print(f'epoch running {epoch}')

    # writer.close()
    torch.save(model.state_dict(), 'transformer_exercise.pth')
