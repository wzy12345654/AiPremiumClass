import pickle
import torch
import json
from torch.utils.data import DataLoader
from process import get_proc, Vocabulary, read_duilian_data
from EncoderDecoderAttenModel import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='runs/duilian_attn')

    # 加载数据
    enc_data, dec_data = read_duilian_data(
        'couplet/train/in.txt',
        'couplet/train/out.txt'
    )

    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)

    with open('vocab.bin','wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab),f)

    with open('encoder.json','w', encoding='utf-8') as f:
        json.dump(enc_data, f)
    with open('decoder.json','w', encoding='utf-8') as f:
        json.dump(dec_data, f)

    dataset = list(zip(enc_data, dec_data))
    dl = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=get_proc(enc_vocab.vocab, dec_vocab.vocab))

    model = Seq2Seq(
        enc_emb_size=len(enc_vocab.vocab),
        dec_emb_size=len(dec_vocab.vocab),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        model.train()
        total_loss = 0
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            enc_input, dec_input, targets = enc_input.to(device), dec_input.to(device), targets.to(device)
            logits, _ = model(enc_input, dec_input)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        writer.add_scalar("Loss/train", total_loss / len(dl), epoch)

    torch.save(model.state_dict(), 'seq2seq_state.bin')
