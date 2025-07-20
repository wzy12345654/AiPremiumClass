import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, maxlen, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1)]
        return self.dropout(x)

# Transformer模型定义
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers,
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout,
                                          batch_first=True)
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        self.predict = nn.Linear(d_model, dec_voc_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        outs = self.transformer(src=enc_emb, tgt=dec_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=enc_pad_mask,
                                tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)

    def encode(self, enc_inp, enc_pad_mask):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb, src_key_padding_mask=enc_pad_mask)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, tgt_mask=dec_mask)

# 数据
class PoemDataset(Dataset):
    def __init__(self, enc_ids, dec_ids):
        self.enc_ids = enc_ids
        self.dec_ids = dec_ids

    def __len__(self):
        return len(self.enc_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.enc_ids[idx]), torch.tensor(self.dec_ids[idx])

def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    enc_pad = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    dec_pad = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    dec_inp = dec_pad[:, :-1]
    dec_tgt = dec_pad[:, 1:]
    return enc_pad, dec_inp, dec_tgt, dec_pad[:, :-1], dec_pad[:, 1:]

def create_mask(dec_inp, enc_pad, dec_pad):
    tgt_len = dec_inp.size(1)
    tgt_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool), diagonal=1)
    enc_pad_mask = enc_pad == 0
    dec_pad_mask = dec_pad == 0
    return tgt_mask, enc_pad_mask, dec_pad_mask


# 推理方法
def generate(model, enc_inp, max_len, stoi, itos, device):
    model.eval()
    enc_pad_mask = (enc_inp == 0)
    memory = model.encode(enc_inp, enc_pad_mask=enc_pad_mask)

    ys = torch.tensor([[stoi['<s>']]], device=device)
    for i in range(max_len):
        tgt_mask = torch.triu(torch.full((ys.size(1), ys.size(1)), float('-inf')), diagonal=1).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = model.predict(out)
        next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == stoi['</s>']:
            break
    return ys[0].tolist()


if __name__ == '__main__':
    raw_corpus = [
    ("春风得意马蹄疾", "一日看尽长安花"),
    ("但愿人长久", "千里共婵娟"),
    ("清风明月本无价", "近水远山皆有情"),
    ("大展宏图兴伟业", "再创辉煌谱新篇"),
    ("一帆风顺年年好", "万事如意步步高"),
    ("风调雨顺年年好", "国泰民安步步高"),
    ("日出江花红胜火", "春来江水绿如蓝"),
    ("万水千山总是情", "风花雪月也传情"),
    ("知足常乐福常在", "淡泊明志寿更长"),
    ("志在四方腾骏马", "心驰万里展雄风"),
    ("海阔凭鱼跃", "天高任鸟飞"),
    ("花好月圆人团圆", "事顺业兴福满堂"),
    ("天增岁月人增寿", "春满乾坤福满门"),
    ("风声雨声读书声声声入耳", "家事国事天下事事事关心"),
    ("心想事成万事顺", "福到运来百业兴"),
    ("神州万里展宏图", "华夏九州铺锦绣"),
    ("大地回春风景丽", "神州奋起气象新"),
    ("春暖花开百业兴", "时来运转千家乐"),
    ("瑞雪兆丰年", "春风迎新岁"),
]
  
    test_sentence = "人生得意须尽欢"

    all_text = ''.join([enc + dec for enc, dec in raw_corpus]) + test_sentence

    chars = sorted(set(all_text))
    special_tokens = ['<pad>', '<s>', '</s>']
    vocab = special_tokens + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    enc_tokens, dec_tokens = [], []
    for i in range(1, len(chars)):
        enc = chars[:i]
        dec = ['<s>'] + chars[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    enc_ids = [[stoi[ch] for ch in tokens] for tokens in enc_tokens]
    dec_ids = [[stoi[ch] for ch in tokens] for tokens in dec_tokens]

    # 数据加载器
    dataset = PoemDataset(enc_ids, dec_ids)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model=256, nhead=8, num_enc_layers=3, num_dec_layers=3,
                               dim_forward=512, dropout=0.1,
                               enc_voc_size=len(stoi), dec_voc_size=len(stoi)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # 模型训练
    for epoch in range(10):
        model.train()
        for enc_pad, dec_inp, dec_tgt, _, _ in loader:
            enc_pad, dec_inp, dec_tgt = enc_pad.to(device), dec_inp.to(device), dec_tgt.to(device)
            tgt_mask, enc_mask, dec_mask = create_mask(dec_inp, enc_pad, dec_inp)
            logits = model(enc_pad, dec_inp, tgt_mask.to(device), enc_mask.to(device), dec_mask.to(device))
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 模型保存
    torch.save(model.state_dict(), 'transformer_couplet.pth')
    print("模型已保存")

    # 测试推理
    test_ids = []
    for ch in test_sentence:
        if ch in stoi:
            test_ids.append(stoi[ch])
        else:
            test_ids.append(stoi['<pad>']) 

    test_ids = torch.tensor([test_ids], device=device)
    pred_ids = generate(model, test_ids, max_len=20, stoi=stoi, itos=itos, device=device)
    pred_text = ''.join([itos[i] for i in pred_ids[1:-1]])  # 去除<s>和</s>
    print(f"上联：{test_sentence}")
    print(f"下联：{pred_text}")