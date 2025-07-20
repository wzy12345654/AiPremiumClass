import re
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def read_data(in_file,out_file):
    """
    读取训练数据返回数据集合
    """
    enc_data, dec_data = [],[]
    in_ = open(in_file, encoding='utf-8')
    out_ = open(out_file, encoding='utf-8')

    for enc, dec in zip(in_, out_):
        #分词
        enc_tks = enc.split()
        dec_tks = dec.split()
        #保存
        enc_data.append(enc_tks)
        dec_data.append(dec_tks)

    #断言
    assert len(enc_data) == len(dec_data), '编码数据与解码数据长度不一致！'

    return enc_data, dec_data

def get_proc(vocab):
    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, lables = [],[],[]
        for enc,dec in data:
            #token -> token index 首尾添加起始和结束token
            enc_idx = [vocab['<s>']] + [vocab[tk] for tk in enc]  +[vocab['</s>']]
            dec_idx = [vocab['<s>']] +[vocab[tk] for tk in dec] + [vocab['</s>']]

            #encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            #decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            #lable
            lables.append(torch.tensor(dec_idx[1:]))

        #数据转换张量 [batch, max_token_len]
        #用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids,batch_first=True)
        dec_input = pad_sequence(dec_ids,batch_first=True)
        targets = pad_sequence(lables,batch_first=True)

        #返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets
    return batch_proc


class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
       
    @classmethod
    def from_file(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().split()
            vocab = ['<pad>'] + [tk for tk in vocab if tk != '']
            # <s>和</s> 作为字符串的起始和结束token
            return cls({tk: i for i,tk in enumerate(vocab)})
        
if __name__ == '__main__':

    #加载词典
    vocab_file = 'couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)

    #训练数据
    enc_train_file = 'couplet/train/in.txt'
    dec_train_file = 'couplet/train/out.txt'

    enc_data, dec_data = read_data(enc_train_file, dec_train_file)

    print('enc length', len(enc_data))
    print('dec length', len(dec_data))
    print('vocab length', len(vocab.vocab))

    #编码 +解码（训练数据）
    dataset = list(zip(enc_data, dec_data))

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=get_proc(vocab.vocab), #回调函数
    )

    #数据缓存
    import json
    with open('couplet/encoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)

    with open('couplet/decoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)

