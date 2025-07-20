import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

current_dir = os.path.dirname(os.path.abspath(__file__))


def build_vocab_from_documents(documents):
    # 字典构建
    no_repeat_tokens = set()
    for cmt in documents:
        no_repeat_tokens.update(cmt.split())

    # set 转换为list,第0个位置添加统一特殊token
    tokens = ["<PAD>", "<UNK>"] + list(no_repeat_tokens)
    vocab = {tk: i for i, tk in enumerate(tokens)}

    return vocab


def build_collate_func(vocab):

    def collate_func(batch):
        comments, labels = [], []
        for item in batch:
            token_index = torch.tensor(
                [vocab[tk] for tk in item[0].split() if tk != " "]
            )
            comments.append(token_index)
            labels.append(item[1])

        # 对评论进行padding
        comments = pad_sequence(comments, batch_first=True, padding_value=0)
        return comments, torch.tensor(labels, dtype=torch.int64)

    return collate_func


if __name__ == "__main__":
    import pickle

    with open(f"{current_dir}/comments_jieba.bin", "rb") as f:
        comments_jieba, labels = pickle.load(f)

    # 构建词典
    vocab = build_vocab_from_documents(comments_jieba)

    # 自定义Dataset处理文本数据转换
    ds = list(zip(comments_jieba, labels))
    dl = DataLoader(ds, batch_size=10, collate_fn=build_collate_func(vocab))

    for item in dl:
        comments, labels = item
        print(comments.shape, labels.shape)
        print(comments[0])
        print(labels[0])
        break
