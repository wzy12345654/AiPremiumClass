import torch
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 构建词典
def build_vocab_from_documents(documents):
    no_repeat_token = set()
    for cmt in documents:
        no_repeat_token.update(cmt.split())
    # 将set转换为list，并在第0个位置添加特殊token
    tokens = ['<PAD>', '<UNK>'] + list(no_repeat_token)

    vocab = { tk:i for i, tk in enumerate(tokens)}
    return vocab


# 给词语进行序列化生成词汇表，并转为张量
def build_collate_func(vocab):
    def collate_func(datch):
        comments, labels = [], []
        for item in datch:
            # 将评论文本分割成单词，并转换为相应的词汇索引序
            token_index = torch.tensor([vocab[tk] for tk in item[0].split() if tk != ''])
            comments.append(token_index)
            labels.append(item[1])

        # padding
        # 对评论序列进行填充，使其长度相同
        comments = pad_sequence(comments , batch_first = True , padding_value = 0)
        return comments , torch.tensor(labels , dtype = torch.int64)
    return collate_func

if __name__ == '__main__':

    # 加载数据
    with open('./data/comments_jieba.bin', 'rb') as f:
        comments_jieba , labels = pickle.load(f)

    # 构建词典
    vocab = build_vocab_from_documents(comments_jieba)

    # 自定义处理文本转换
    ds = list(zip(comments_jieba , labels))

    # 数据加载
    dl = DataLoader(ds, batch_size = 10, collate_fn = build_collate_func(vocab), shuffle = True)
    for item in dl:
        print(item[0], item[1])
        break


