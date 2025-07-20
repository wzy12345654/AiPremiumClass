import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json


def read_data(in_file, out_file):
    """
    读取训练数据并返回数据集合。
    该函数会从指定的输入文件和输出文件中读取数据，对每行数据进行分词处理，
    并将分词结果分别存储到编码器数据列表和解码器数据列表中。最后会检查
    两个列表的长度是否一致，若不一致则抛出断言错误。

    Args:
        in_file (str): 包含编码器输入数据的文件路径。
        out_file (str): 包含解码器输出数据的文件路径。

    Returns:
        tuple: 包含两个列表的元组，第一个列表为编码器数据，第二个列表为解码器数据。
    """

    # 初始化编码器数据列表和解码器数据列表
    enc_data, dec_data = [], []
    # 打开包含编码器输入数据的文件
    in_ = open(in_file)
    # 打开包含解码器输出数据的文件
    out_ = open(out_file)

    # 逐行同时读取两个文件的数据
    for enc, dec in zip(in_, out_):
        # 对编码器输入数据行进行分词处理，按空格分割成单词列表
        enc_tks = enc.split()
        # 对解码器输出数据行进行分词处理，按空格分割成单词列表
        dec_tks = dec.split()
        # 将分词后的编码器输入数据添加到编码器数据列表中
        enc_data.append(enc_tks)
        # 将分词后的解码器输出数据添加到解码器数据列表中
        dec_data.append(dec_tks)

    # 断言检查编码器数据列表和解码器数据列表的长度是否一致
    # 若不一致，抛出 AssertionError 异常并显示提示信息
    assert len(enc_data) == len(dec_data), '编码数据与解码数据长度不一致！'

    # 返回包含编码器数据列表和解码器数据列表的元组
    return enc_data, dec_data

def get_proc(vocab):
    """
    创建一个批次数据处理函数，用于将输入的批次数据转换为模型所需的张量格式。

    Args:
        vocab (dict): 词汇表，将 token 映射为对应的索引。

    Returns:
        callable: 用于处理批次数据的函数 batch_proc。
    """
    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        对输入的批次数据进行处理，将文本数据转换为张量，并进行填充操作。

        Args:
            data (list): 一个批次的数据，每个元素是一个元组，包含编码器输入和解码器输入。

        Returns:
            tuple: 包含三个张量的元组，分别为编码器输入、解码器输入和目标标签。
        """
        # 初始化编码器输入索引列表、解码器输入索引列表和目标标签列表
        enc_ids, dec_ids, labels = [],[],[]
        for enc, dec in data:
            # 将编码器输入的 token 转换为对应的索引，并在首尾添加起始和结束 token
            enc_idx = [vocab['<s>']] + [vocab[tk] for tk in enc] + [vocab['</s>']]
            # 将解码器输入的 token 转换为对应的索引，并在首尾添加起始和结束 token
            dec_idx = [vocab['<s>']] + [vocab[tk] for tk in dec] + [vocab['</s>']]

            # encoder_input: 将编码器输入索引列表转换为张量并添加到 enc_ids 列表中
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input: 取解码器输入索引列表除最后一个元素外的部分，转换为张量并添加到 dec_ids 列表中
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label: 取解码器输入索引列表除第一个元素外的部分，转换为张量并添加到 labels 列表中
            labels.append(torch.tensor(dec_idx[1:]))
        
        # 数据转换为张量，形状为 [batch, max_token_len]
        # 使用批次中最长的 token 序列构建张量，对较短的序列进行填充
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        # 返回模型训练和推理所需的数据
        return enc_input, dec_input, targets

    # 返回回调函数，用于在 DataLoader 中处理批次数据
    return batch_proc    

class Vocabulary:

    def __init__(self, vocab):
        """
        初始化 Vocabulary 类的实例。

        Args:
            vocab (dict): 一个字典，将 token 映射为对应的整数索引。
                          该字典通常包含特殊 token，如 '<pad>'、'<s>' 和 '</s>'。
        """
        # 将传入的词汇表字典赋值给实例属性 self.vocab
        self.vocab = vocab

    @classmethod
    def from_file(cls, vocab_file):
        """
        从文件中读取词汇表并创建 Vocabulary 类的实例。

        该类方法会打开指定的词汇表文件，逐行读取文件内容，将其转换为词汇列表，
        并添加特殊 token '<pad>'，最后将词汇列表转换为字典，键为词汇，值为对应的索引，
        并使用该字典创建 Vocabulary 类的实例。

        Args:
            cls (class): 类本身，在类方法中作为第一个参数。
            vocab_file (str): 词汇表文件的路径。

        Returns:
            Vocabulary: Vocabulary 类的实例，包含从文件中读取并处理后的词汇表。
        """
        # 打开词汇表文件，使用 utf-8 编码
        with open(vocab_file, encoding='utf-8') as f:
            # 读取文件内容并按换行符分割成词汇列表
            vocab = f.read().split('\n')
            # 在词汇列表开头添加特殊 token '<pad>'，并过滤掉空字符串
            vocab = ['<pad>'] + [tk for tk in vocab if tk != '']
            # <s>和</s> 作为字符串的起始和结束token
            # 创建一个字典，将每个词汇映射到其在列表中的索引
            vocab_dict = {tk: i for i, tk in enumerate(vocab)}
            # 使用创建好的词汇表字典创建 Vocabulary 类的实例并返回
            return cls(vocab_dict)

def run_process():
    # 加载词典
    # 定义词汇表文件的路径
    vocab_file = 'week8_codes/couplet/vocabs'
    # 调用 Vocabulary 类的 from_file 方法从文件中读取词汇表并创建实例
    vocab = Vocabulary.from_file(vocab_file)

    # 训练数据
    # 定义编码器训练数据文件的路径
    enc_train_file = 'week8_codes/couplet/train/in.txt'
    # 定义解码器训练数据文件的路径
    dec_train_file = 'week8_codes/couplet/train/out.txt'

    # 调用 read_data 函数读取训练数据，返回编码器数据和解码器数据
    enc_data, dec_data = read_data(enc_train_file, dec_train_file)
    
    # 打印编码器数据的长度
    print('enc length', len(enc_data))
    # 打印解码器数据的长度
    print('dec length', len(dec_data))

    # 打印词汇表中词汇的数量
    print('词汇数量', len(vocab.vocab))
    
    # 编码+解码（训练样本）
    # 将编码器数据和解码器数据组合成一个列表，每个元素是一个包含编码器输入和解码器输入的元组
    dataset = list(zip(enc_data, dec_data))
    # Dataloader
    # 创建一个 DataLoader 对象，用于批量加载数据
    dataloader = DataLoader(
        dataset,  # 要加载的数据集
        batch_size=2,  # 每个批次包含的样本数量
        shuffle=True,  # 是否在每个 epoch 开始时打乱数据顺序
        collate_fn=get_proc(vocab.vocab)  # 自定义的批次数据处理函数
    )

    # 数据缓存
    # 此处可添加数据缓存相关代码，当前为空

    # 数据整体json数据集（json）
    # 将编码器数据以 JSON 格式写入文件
    with open('week8_codes/encoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)  
    
    # 将解码器数据以 JSON 格式写入文件
    with open('week8_codes/decoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)  


if __name__ == '__main__':
    run_process()
   

