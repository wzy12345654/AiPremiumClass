import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import pickle
import random

# 准备数据
def read_data(data_file , sample_size):

    # 准备编码器与解码器的数据
    data1 = []
    with open(data_file , 'r' , encoding = 'utf-8') as f:
        lines = [line for line in f.readlines() if line.strip("\n") if line.strip() != '']
        # 随机采样一万行
        if len(lines) > sample_size:
            lines = random.sample(lines, sample_size)

        # 读取每一行数据
        for line in lines:
            if line == '':
                continue
            # 数据清洗
            line = line.replace('，','').replace(' ','')
            # 分词
            data = jieba.lcut(line)
            # 给解码器添加起始字符
            if data_file == '../data/out.txt':
                data = ['BOS'] + data + ['EOS']
            # 保存
            data1.append(data)
    return data1

def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]
            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)
        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets
    # 返回回调函数
    return batch_proc

# 构建词汇表
class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
    @classmethod
    def from_documents(cls, documents):
        # 字典构建（字符为token、词汇为token）
        no_repeat_tokens = set()
        for cmt in documents:
            no_repeat_tokens.update(list(cmt))  # token list
        # set转换为list，第0个位置添加统一特殊token
        tokens = ['PAD','UNK'] + list(no_repeat_tokens)
        vocab = { tk:i for i, tk in enumerate(tokens)}
        return cls(vocab)


if __name__ == '__main__':

    # 读取数据
    dec_data = read_data('../data/out.txt' , sample_size = 10000)
    enc_data = read_data('../data/in.txt' , sample_size = 10000)

    print('enc length', len(enc_data))
    print('dec length', len(dec_data))
    # 构建词汇表
    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)
    print('编码器词汇数量', len(enc_vocab.vocab))
    print('解码器词汇数量', len(dec_vocab.vocab))

    # 将数据压缩成训练样本
    train_data = list(zip(enc_data, dec_data))
    # 构建批次数据处理函数
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）
    dataloader = DataLoader(
        train_data,
        batch_size = 16,
        shuffle = True,
        collate_fn = get_proc(enc_vocab.vocab, dec_vocab.vocab)
    )

    # 数据整体json数据集（json）
    with open('cencoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)

    with open('cdecoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)

    with open('cvocab.bin', 'wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab), f)



