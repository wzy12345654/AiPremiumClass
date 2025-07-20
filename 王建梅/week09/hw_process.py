import os
import json
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids,labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx))
            # dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        
        # 数据转换张量 [batch, max_token_len]
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        #targets = pad_sequence(labels, batch_first=True)

        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input

    # 返回回调函数
    return batch_proc    

def get_word_list(filepath,is_decode=False):
    # 读取训练数据返回数据集合
    data_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            # 空格分隔的字符串转list
            word_tokens = line.split(' ')
            # 去除最后一个空字符串
            if word_tokens[-1] == '':
                word_tokens = word_tokens[:-1]
            # 解码，开头和结尾添加'<s>'和'</s>'
            if is_decode:
                word_tokens = ['<s>'] + word_tokens + ['</s>']
            data_list.append(word_tokens)
    return data_list

def read_data():
    # 读取训练数据，返回数据集合
    train_data,test_data = [],[]
    train_enc_data = get_word_list('data/couplet/train/in.txt')
    train_dec_data = get_word_list('data/couplet/train/out.txt',is_decode=True)
    test_enc_data = get_word_list('data/couplet/test/in.txt')
    test_dec_data = get_word_list('data/couplet/test/out.txt',is_decode=True)
    # 编码与解码的数据集
    train_data = list(zip(train_enc_data,train_dec_data))
    test_data = list(zip(test_enc_data,test_dec_data))
    return train_data,test_data

def generate_square_subsequent_mask(sz):
    # 生成上三角掩码矩阵, 用于屏蔽后面位置Token的损失, 形状为[sz, sz]
    mask = (torch.triu(torch.ones((sz, sz), dtype=torch.bool)) == 1).transpose(0, 1)
    # 把填充位置的掩码值设为-inf 极小值
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    # 生成掩码矩阵
    src_seq_len = src.shape[1]  # 源序列长度，形状为[batch_size, src_seq_len]
    tgt_seq_len = tgt.shape[1]  # 目标序列长度，形状为[batch_size, tgt_seq_len]

    # 生成源序列的掩码矩阵, 全0矩阵, 形状为[batch_size, src_seq_len, src_seq_len]
    #src_mask = torch.triu(torch.ones((src_seq_len, src_seq_len), dtype=torch.bool), diagonal=1)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)
    # 生成目标序列的下三角掩码矩阵,保证解码器在预测时只能看到之前的位置, 形状为[batch_size, tgt_seq_len, tgt_seq_len]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 生成源序列的padding掩码矩阵, 用于屏蔽填充位置的损失, 形状为[batch_size, src_seq_len]
    src_padding_mask = (src == 0)
    # 生成目标序列的padding掩码矩阵, 用于屏蔽填充位置的损失, 形状为[batch_size, tgt_seq_len]
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class Vocabulary:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_file(cls,vocab_file):
        # 字典构建（字符为token、词汇为token）
        # set转换为list，第0个位置添加统一特殊token
        with open(vocab_file, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            vocab_list = ['<pad>'] + [tk for tk in lines if tk != '']
            # <s>和</s> 作为字符串的起始和结束token
            vocab = {tk:i for i, tk in enumerate(vocab_list)}
        return cls(vocab)

if __name__ == '__main__':
    # 加载词典
    vocab_file = 'data/couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)
    train_data,test_data = read_data()
    # 将 train_data,test_data,vocab 数据集和字典保存起来
    with open('data/model/hw_train_dataset.pkl','wb') as f:
        pickle.dump(train_data,f)
    with open('data/model/hw_test_dataset.pkl','wb') as f:
        pickle.dump(test_data,f)
    with open('data/model/hw_vocab.bin','wb') as f:
        pickle.dump(vocab.vocab,f)