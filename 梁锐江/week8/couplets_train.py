import pickle
import json

from torch import nn
from torch.utils.data import DataLoader
from chinese_couplets import BatchDataProcessor
from torch.utils.tensorboard import SummaryWriter
import torch

from 梁锐江.week8.couplets_model import CoupletsSeq2SeqModel

if __name__ == '__main__':
    epochs = 3
    embedding_dim = 512
    hidden_dim = 128

    writer = SummaryWriter("couplets_train")

    with open('couple_vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    with open('couple_encode.json', 'r', encoding='utf-8') as f:
        enc_docs = json.load(f)

    with open('couple_decode.json', 'r', encoding='utf-8') as f:
        dec_docs = json.load(f)

    processor = BatchDataProcessor(enc_vocab, dec_vocab)
    dataset = list(zip(enc_docs, dec_docs))
    batch_data = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=processor.get_word_idx)

    model = CoupletsSeq2SeqModel(len(enc_vocab), len(dec_vocab), embedding_dim, hidden_dim)
    model.to('cuda')
    loss_fc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_step = 0
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, targets in batch_data:
            optimizer.zero_grad()
            enc_inputs = enc_inputs.to('cuda')
            dec_inputs = dec_inputs.to('cuda')
            targets = targets.to('cuda')

            logits, hidden = model(enc_inputs, dec_inputs)
            loss = loss_fc(logits.view(-1, len(dec_vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_step += 1
            writer.add_scalar('loss', loss, total_step)
        print(f"{epoch} running")

    writer.close()
    torch.save(model.state_dict(), 'couplets_state_attention.bin')
