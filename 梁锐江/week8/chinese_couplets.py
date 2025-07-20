import matplotlib.pyplot as plt
import json
import pickle
import torch.nn.utils.rnn as rnn

import torch


class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def build_docs_vocab(cls, docs):
        tokens_set = set()
        for doc in docs:
            tokens_set.update(doc)

        vocab = {token: idx for idx, token in enumerate(tokens_set)}
        return cls(vocab)


class BatchDataProcessor:
    def __init__(self, enc_vocab, dec_vocab):
        self.enc_vocab = enc_vocab
        self.dec_vocab = dec_vocab

    def get_word_idx(self, batch_data):
        enc_inputs, dec_inputs, targets = [], [], []
        for enc_doc, dec_doc in batch_data:
            enc_idx = [self.enc_vocab.get(token, 0) for token in enc_doc]
            dec_idx = [self.dec_vocab.get(token, 0) for token in dec_doc]

            enc_inputs.append(torch.tensor(enc_idx))
            dec_inputs.append(torch.tensor(dec_idx[:-1]))
            targets.append(torch.tensor(dec_idx[1:]))

        enc_inputs = rnn.pad_sequence(enc_inputs, batch_first=True)
        dec_inputs = rnn.pad_sequence(dec_inputs, batch_first=True)
        targets = rnn.pad_sequence(targets, batch_first=True)

        return enc_inputs, dec_inputs, targets


def data_process():
    with open('./data/in.txt', 'r', encoding='utf-8') as in_file:
        in_lines = in_file.read().split("\n")
        enc_docs = [list(line) for line in in_lines]

    with open('./data/out.txt', 'r', encoding='utf-8') as out_file:
        out_lines = out_file.read().split("\n")
        out_docs = [['BOS'] + list(line) + ['EOS'] for line in out_lines]

    return enc_docs, out_docs


"""
    数据分析
"""


def data_analys(enc_docs, out_docs):
    enc_length = [len(line) for line in enc_docs]
    out_length = [len(line) for line in out_docs]

    # 绘制箱线图（将两组数据放在同一图中）
    plt.boxplot(
        [enc_length, out_length],
        labels=['in', 'out'],  # 设置分组标签
        patch_artist=True,  # 允许填充箱体颜色
        showmeans=True,  # 显示均值线
        boxprops={'facecolor': 'lightblue', 'color': 'darkblue'},  # 箱体样式
        medianprops={'color': 'red'},  # 中位数线样式
        meanprops={'marker': 'D', 'markerfacecolor': 'gold'}  # 均值标记样式
    )

    # 添加标题和标签
    plt.title('in vs out', fontsize=14)
    plt.ylabel('len', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)  # 添加横向网格线

    # 显示图表
    plt.tight_layout()  # 自动调整布局
    plt.show()


if __name__ == '__main__':
    enc_docs, out_docs = data_process()

    enc_vocab = Vocabulary.build_docs_vocab(enc_docs)
    dec_vocab = Vocabulary.build_docs_vocab(out_docs)

    # 数据集缓存
    with open('couple_encode.json', 'w', encoding='utf-8') as f:
        json.dump(enc_docs, f)

    with open('couple_decode.json', 'w', encoding='utf-8') as f:
        json.dump(out_docs, f)

    # 词典表保存
    with open('couple_vocab.bin', 'wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab), f)
