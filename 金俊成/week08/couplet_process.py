import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def read_couplet_data(in_file, out_file):
    """
    读取对联数据集，返回上联和下联数据
    """
    enc_data, dec_data = [], []
    
    # 读取上联数据
    with open(in_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                enc_data.append(['BOS'] + list(line) + ['EOS'])
    
    # 读取下联数据
    with open(out_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dec_data.append(['BOS'] + list(line) + ['EOS'])

    assert len(enc_data) == len(dec_data), '上联与下联数据长度不一致！'
    return enc_data, dec_data

def get_proc(enc_voc, dec_voc):
    def batch_proc(data):
        """
        批次数据处理
        """
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        # 数据转换张量 [batch, max_token_len]
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        return enc_input, dec_input, targets

    return batch_proc

class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def from_documents(cls, documents):
        # 构建词表
        tokens = set()
        for doc in documents:
            tokens.update(doc)
        
        # 添加特殊token
        vocab_list = ['PAD', 'UNK', 'BOS', 'EOS'] + list(tokens)
        vocab = {tk: i for i, tk in enumerate(vocab_list)}
        
        return cls(vocab)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, token):
        return self.vocab.get(token, self.vocab['UNK'])

    def decode(self, indices):
        return [self.inv_vocab[idx] for idx in indices]