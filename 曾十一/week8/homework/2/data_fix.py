import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def read_data(data_file_in, data_file_out, vocab_file):
    """
    读取训练数据返回数据集合
    """

    enc_data, dec_data ,vocab_data= [], [] , []

    with open(data_file_in, encoding='utf-8') as f:
        # 读取记录行，去掉首尾空白，并按行分割
        lines = f.read().strip().split('\n')

        for line in lines:
            if line.strip() == '':
                continue
            # 分词
            enc_tks = line.strip().split()
            # 保存
            enc_data.append(enc_tks)

    with open(data_file_out, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

        for line in lines:
            if line.strip() == '':
                continue
            dec_tks = line.strip().split()
            dec_tks = ['<s>'] + dec_tks + ['</s>']
            dec_data.append(dec_tks)

    with open(vocab_file, encoding='utf-8') as f:
        # 读取记录行，去掉首尾空白，并按行分割
        lines = f.read().strip().split('\n')

        for line in lines:
            if line.strip() == '':
                continue
            # 保存
            vocab_data.append(line.strip())



    # 保守起见，增加断言，保证数据一一对应
    assert len(enc_data) == len(dec_data), f'编码数据与解码数据长度不一致！enc:{len(enc_data)}, dec:{len(dec_data)}'

    return enc_data, dec_data, vocab_data

def get_proc(vocab):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [vocab[tk] for tk in enc]
            dec_idx = [vocab[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        
        # 数据转换张量 [batch, max_token_len]
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets

    # 返回回调函数
    return batch_proc    

class Vocabulary:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        # 字典构建（字符为token、词汇为token）
        no_repeat_tokens = set()
        for cmt in documents:
            no_repeat_tokens.add(cmt)  # token list
        # set转换为list，第0个位置添加统一特殊token
        tokens = ['PAD','UNK'] + list(no_repeat_tokens)

        vocab = { tk:i for i, tk in enumerate(tokens)}

        return cls(vocab)

if __name__ == '__main__':
    
    enc_data, dec_data, vocab_data = read_data(r'/mnt/data_1/zfy/4/week8/资料/homework/couplet/train/in.txt', 
                                  r'/mnt/data_1/zfy/4/week8/资料/homework/couplet/train/out.txt',
                                  r'/mnt/data_1/zfy/4/week8/资料/homework/couplet/vocabs.txt')
    
    print('enc length', len(enc_data))
    print('dec length', len(dec_data))

    vocab = Vocabulary.from_documents(vocab_data)
    
    # 编码+解码（训练样本）
    dataset = list(zip(enc_data, dec_data))
    # Dataloader

    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=get_proc(vocab.vocab)   # callback
    )

    # 数据缓存
    import json

    # 数据整体json数据集（json）
    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework_1/encoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)  
    
    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework_1/decoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)  

    import pickle
    with open(r'/mnt/data_1/zfy/4/week8/资料/homework/homework_1/vocab.bin','wb') as f:
        pickle.dump((vocab.vocab),f)


    # # 数据每行都是json数据（jsonl）
    # with open('encoders.json', 'w', encoding='utf-8') as f:
    #     for enc in enc_data:
    #         str_json = json.dumps(enc)
    #         f.write(str_json + '\n')
