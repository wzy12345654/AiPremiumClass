from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import json
import pickle

def read_data(in_file, out_file):

    enc_data, dec_data = [], []

    with open(in_file, "r", encoding="utf-8") as f:
        enc_lines = f.read().split("\n")
    with open(out_file, "r", encoding="utf-8") as f:
        dec_lines = f.read().split("\n")

    # enc_lines 与 dec_lines行数相同
    # print(len(enc_lines), len(dec_lines))

    for i in range(len(enc_lines)):
        # 去除标点和空格
        enc_lines[i] = enc_lines[i].replace("。", "").replace("，", "").replace("！", "").replace("？", "").replace(" ", "")
        dec_lines[i] = dec_lines[i].replace("。", "").replace("，", "").replace("！", "").replace("？", "").replace(" ", "")
        # 分词
        enc_tks = list(enc_lines[i])
        dec_tks = ['BOS'] + list(dec_lines[i]) + ['EOS']

        # 保存
        enc_data.append(enc_tks)
        dec_data.append(dec_tks)
        assert len(enc_data) == len(dec_data)

    return enc_data, dec_data

def get_proc(enc_vocab, dec_vocab):
    def batch_proc(data):
        """
        批次数据处理函数
        """
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in data:
            # token -> index
            enc_idx = [enc_vocab.get(word, enc_vocab['UNK']) for word in enc]
            dec_idx = [dec_vocab.get(word, dec_vocab['UNK']) for word in dec]

            enc_ids.append(torch.tensor(enc_idx))
            dec_ids.append(torch.tensor(dec_idx[:-1]))  # 去掉最后一个EOS
            labels.append(torch.tensor(dec_idx[1:]))  # 去掉第一个BOS

        # padding
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        labels = pad_sequence(labels, batch_first=True)

        # 返回填充后的数据
        return enc_input, dec_input, labels
    # 返回回调函数
    return batch_proc

class Vocabulary:
    """
    词汇表类
    """
    def __init__(self, vocab):
        self.vocab = vocab
    
    @classmethod
    def build_from_doc(cls, docs):
        vocab_set = set()
        for cmt in docs:
            vocab_set.update(list(cmt))
        vocab_set = ['PAD', 'UNK'] + list(vocab_set)  # PAD:padding, UNK: unknown
        # 构建词汇到索引的映射
        vocab = {word: i for i, word in enumerate(vocab_set)}
        return cls(vocab)
    
if __name__ == "__main__":
    # 读取数据
    enc_data, dec_data = read_data("couplet/train/in.txt", "couplet/train/out.txt")

    enc_vocab = Vocabulary.build_from_doc(enc_data)
    dec_vocab = Vocabulary.build_from_doc(dec_data)

    print(f"编码器词汇 size: {len(enc_vocab.vocab)}")
    print(f"解码器词汇 size: {len(dec_vocab.vocab)}")

    # 编码+解码
    dataset = list(zip(enc_data, dec_data))

    dataloader = DataLoader(
        dataset,
        batch_size = 4,
        shuffle = True,
        collate_fn = get_proc(enc_vocab.vocab, dec_vocab.vocab)
    )

    # 测试dataloader
    # for enc_data, dec_data, labels in dataloader:
    #     print("test dataloader")
    #     break

    # 保存enc_data和dec_data
    with open("couplet/encoder.json", "w", encoding="utf-8") as f:
        json.dump(enc_data, f)
    
    with open("couplet/decoder.json", "w", encoding="utf-8") as f:
        json.dump(dec_data, f)
    
    # 保存词典
    with open("couplet/vocab.pkl", "wb") as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab), f)