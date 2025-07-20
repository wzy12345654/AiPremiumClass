import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import math

# ================= 位置编码 =================
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)

        pos_embdding = pos_embdding.unsqueeze(0)  # ✅ 正确的位置
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)
        return self.dropout(self.pos_embedding[:, :token_len, :] + token_embdding)



# ================= 自定义 Transformer =================
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout, batch_first=True)
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        self.predict = nn.Linear(d_model, dec_voc_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=enc_pad_mask, 
                                tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)

    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)

# ================= 主程序 =================
if __name__ == '__main__':
    corpus = "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    enc_tokens, dec_tokens = [], []

    for i in range(1, len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    special_tokens = ['<pad>', '<s>', '</s>']
    vocab = {ch: i + len(special_tokens) for i, ch in enumerate(sorted(set(chs)))}
    for t in special_tokens:
        vocab[t] = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    pad_id = vocab['<pad>']
    s_id = vocab['<s>']
    e_id = vocab['</s>']

    def tokens_to_ids(tokens):
        return [vocab[t] for t in tokens]

    class MyDataset(Dataset):
        def __init__(self, enc_tokens, dec_tokens):
            self.enc = [torch.tensor(tokens_to_ids(x)) for x in enc_tokens]
            self.dec = [torch.tensor(tokens_to_ids(x)) for x in dec_tokens]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, idx):
            return self.enc[idx], self.dec[idx]

    def collate_batch(batch):
        enc, dec = zip(*batch)
        enc_pad = pad_sequence(enc, batch_first=True, padding_value=pad_id)
        dec_pad = pad_sequence(dec, batch_first=True, padding_value=pad_id)

        dec_input = dec_pad[:, :-1]
        dec_target = dec_pad[:, 1:]

        tgt_len = dec_input.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)

        enc_pad_mask = (enc_pad == pad_id)
        dec_pad_mask = (dec_input == pad_id)

        return enc_pad, dec_input, dec_target, tgt_mask, enc_pad_mask, dec_pad_mask

    dataset = MyDataset(enc_tokens, dec_tokens)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

    model = Seq2SeqTransformer(d_model=128, nhead=4, num_enc_layers=2, num_dec_layers=2,
                               dim_forward=128, dropout=0.1,
                               enc_voc_size=len(vocab), dec_voc_size=len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(200):
        total_loss = 0
        for enc_pad, dec_in, dec_target, tgt_mask, enc_pad_mask, dec_pad_mask in loader:
            enc_pad, dec_in, dec_target = enc_pad.to(device), dec_in.to(device), dec_target.to(device)
            tgt_mask = tgt_mask.to(device)
            enc_pad_mask = enc_pad_mask.to(device)
            dec_pad_mask = dec_pad_mask.to(device)

            output = model(enc_pad, dec_in, tgt_mask, enc_pad_mask, dec_pad_mask)
            loss = criterion(output.view(-1, output.size(-1)), dec_target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "my_transformer.pth")

    # 推理函数
    def greedy_decode(model, enc_input, max_len=10):
        model.eval()
        enc_input = torch.tensor(tokens_to_ids(list(enc_input))).unsqueeze(0).to(device)
        memory = model.encode(enc_input)

        ys = torch.tensor([[s_id]], dtype=torch.long).to(device)

        for i in range(max_len):
            tgt_mask = torch.triu(torch.ones(ys.size(1), ys.size(1)) * float('-inf'), diagonal=1).to(device)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.predict(out[:, -1, :])
            next_word = torch.argmax(prob, dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_word]]).to(device)], dim=1)
            if next_word == e_id:
                break

        return ''.join([inv_vocab[i] for i in ys[0].cpu().tolist()[1:-1]])

    model.load_state_dict(torch.load("my_transformer.pth"))
    print(greedy_decode(model, "人生得意"))
