import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from config import config  

class CoupletDataset(Dataset):
    """对联数据集类，处理数据的加载和预处理"""
    def __init__(self, in_path, out_path, vocab=None, max_len=100):
        # 读取原始数据
        self.in_data = open(in_path, encoding='utf-8').read().splitlines()
        self.out_data = open(out_path, encoding='utf-8').read().splitlines()
        self.pairs = list(zip(self.in_data, self.out_data))
        
        # 构建加载词汇表
        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab
            self.rev_vocab = {v:k for k,v in vocab.items()}
        
        self.max_len = max_len

    def build_vocab(self):
        """根据训练数据构建词汇表"""
        counter = Counter()
        for line in self.in_data + self.out_data:
            counter.update(line.split())  # 按空格分词
        
        # 初始化基础词汇
        vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
        # 添加高频词
        for idx, (char, _) in enumerate(counter.most_common(config['max_vocab_size']-4)):
            vocab[char] = idx + 4
        
        self.vocab = vocab
        self.rev_vocab = {v:k for k,v in vocab.items()}

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """处理单个样本，添加特殊标记并转换为索引"""
        src = ['<sos>'] + self.pairs[idx][0].split() + ['<eos>']
        trg = ['<sos>'] + self.pairs[idx][1].split() + ['<eos>']
        
        # 将词转换为索引，未知词用<unk>表示
        src_ids = [self.vocab.get(word, 3) for word in src]
        trg_ids = [self.vocab.get(word, 3) for word in trg]
        
        return (torch.LongTensor(src_ids).to(config['device']), 
                torch.LongTensor(trg_ids).to(config['device']))

def collate_fn(batch):
    """批处理函数，用于填充和长度排序"""
    src_batch, trg_batch = zip(*batch)
    
    # 获取各序列实际长度
    src_lens = [len(x) for x in src_batch]
    
    # 填充序列
    src_pad = pad_sequence(src_batch, padding_value=0).to(config['device'])
    trg_pad = pad_sequence(trg_batch, padding_value=0).to(config['device'])
    
    return (src_pad, 
           torch.tensor(src_lens, dtype=torch.long).to(config['device']),  # 添加数据类型
           trg_pad)