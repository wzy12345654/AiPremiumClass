import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import load_data, Vocab, CoupletDataset, get_collate_fn
from model import Seq2Seq
from config import Config
import torch.nn as nn
import pickle


def main():
    config = Config()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    enc_texts, dec_texts = load_data(
        config.enc_data_path,
        config.dec_data_path
    )

    # 构建词汇表
    enc_vocab = Vocab.build_from_texts(enc_texts)
    dec_vocab = Vocab.build_from_texts(dec_texts)

    # 保存词汇表
    with open(config.enc_vocab_path, 'wb') as f:
        pickle.dump(enc_vocab, f)
    with open(config.dec_vocab_path, 'wb') as f:
        pickle.dump(dec_vocab, f)

    # 数据集
    dataset = CoupletDataset(enc_texts, dec_texts)
    collate_fn = get_collate_fn(enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=collate_fn)

    # 模型初始化
    model = Seq2Seq(config, enc_vocab, dec_vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dec_vocab.stoi[dec_vocab.pad])

    # TensorBoard
    writer = SummaryWriter()

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batches, (src, trg, target) in enumerate(dataloader):
            src, trg, target = src.to(device), trg.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            loss = criterion(output.view(-1, output.size(-1)),
                             target.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            total_loss += loss.item()
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(dataloader) + batches)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

    # 保存模型
    torch.save(model.state_dict(), config.model_save_path)
    writer.close()


if __name__ == "__main__":
    main()
