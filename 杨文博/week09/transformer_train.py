from transformer_vocabulary import Vocabulary
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformer_vocabulary import Config
from transformer_model import PoetryTransformer
import torch.optim as optim
from tqdm import tqdm
import pickle


def prepare4dataloader(vocab: Vocabulary):
    encoder_sequence = [[vocab.word2idx.get(word, 0) for word in scentence] for scentence in vocab.upper_couplets]
    decoder_input_sequence = [[vocab.word2idx['<bos>']] + [vocab.word2idx.get(word, 0) for word in scentence] for
                              scentence in vocab.lower_couplets]
    decoder_output_sequence = [[vocab.word2idx.get(word, 0) for word in scentence] + [vocab.word2idx['<eos>']] for
                               scentence in vocab.lower_couplets]
    data_set = [
        (torch.tensor(u), torch.tensor(l), torch.tensor(p))
        for u, l, p in zip(encoder_sequence, decoder_input_sequence, decoder_output_sequence)
    ]
    return data_set


def generate_masks(src, tgt_input, pad_idx):
    src_key_padding_mask = (src == pad_idx)  # [batch, src_len]
    tgt_seq_len = tgt_input.size(1)
    tgt_mask = torch.triu(
        torch.full((tgt_seq_len, tgt_seq_len), float('-inf'), device=src.device),
        diagonal=1
    )
    return tgt_mask, src_key_padding_mask


def collate_fn(batch):
    # 获取当前 batch 中的上联和下联
    encoder_inputs, decoder_inputs, decoder_outputs = zip(*batch)

    # 对上联和下联进行填充，使它们的长度与当前 batch 中的最大长度相同
    encoder_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=1)  # padding_value为<pad>的索引
    decoder_input_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=1)
    decoder_output_padded = pad_sequence(decoder_outputs, batch_first=True, padding_value=1)

    return encoder_padded, decoder_input_padded, decoder_output_padded


def train(model: nn.Module,
          device: torch.device,
          data_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          epochs: int):
    model.train()
    all_epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for enc_input, dec_input, dec_output in pbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            dec_output = dec_output.to(device)

            optimizer.zero_grad()
            tgt_mask, src_key_padding_mask = generate_masks(enc_input, dec_input, 0)
            # output: [batch_size, seq_len, vocab_size]
            output = model(enc_input, dec_input,
                           tgt_mask=tgt_mask,
                           enc_padding_mask=src_key_padding_mask)  # [batch_size, seq_len, vocab_size]

            # Flatten for loss computation
            output = output.view(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
            dec_output = dec_output.view(-1)  # [batch_size * seq_len]

            loss = criterion(output, dec_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=loss.item())

        average_loss = total_loss / batch_count
        all_epoch_losses.append(average_loss)

        print(f"[Epoch {epoch + 1}] Loss: {average_loss:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("./vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    data_set = prepare4dataloader(vocab)
    data_loader = DataLoader(data_set, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
    model = PoetryTransformer(len(vocab.word2idx), len(vocab.word2idx), Config.embedding_dim, Config.num_heads,
                              Config.num_layers, Config.num_layers,
                              Config.dropout, Config.max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train(model, device, data_loader, optimizer, criterion, Config.epoches)
    torch.save(model.state_dict(), 'couplet_transformer.pt')
    print("模型训练完成并保存！")

if __name__ == '__main__':
    main()