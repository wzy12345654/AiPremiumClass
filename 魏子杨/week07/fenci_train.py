import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba

# 构建词汇表，即每个单词有对应的数字
def build_from_doc(doc):
    #去除重复的词
    vocab = set()
    for line in doc:
        vocab.update(line[0])
    # 排序，以坐标作为对应向量，未收入的词给'UNK'，长度不够的给PAD，对应下标0
    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx
    # 自定义数据转换方法(callback function)回调函数
    # 该函数会在每个batch数据加载时被调用
def convert_data(batch_data):
    comments, votes = [],[]
    # 分别提取评论和标签
    for comment, vote in batch_data:
        comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
        votes.append(vote)
    
    # 将评论和标签转换为tensor
    commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])  # 填充为相同长度
    labels = torch.tensor(votes)
    # 返回评论和标签
    return commt, labels
    
class Comments_Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
        return output

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练语料
    with open('week7/comments_jieba.pkl','rb') as f:
        comments_data, labels = pickle.load(f)

    # 构建词汇表
    vocab = build_from_doc(comments_data)
    print('词汇表大小:', len(vocab))

    # 所有向量集合 Embedding（词嵌入），每句话做成对应的100长度的向量
    emb = nn.Embedding(len(vocab), 100) # 词汇表大小，向量维度

    train_ds = list(zip(comments_data, labels))
    # 通过Dataset构建DataLoader
    dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, 
                            collate_fn=convert_data)

    # 构建模型
    # vocab_size: 词汇表大小
    # embedding_dim: 词嵌入维度
    # hidden_size: LSTM隐藏层大小
    # num_classes: 分类数量
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (cmt, lbl) in enumerate(dataloader):
            cmt = cmt.to(device)
            lbl = lbl.to(device)

            # 前向传播
            outputs = model(cmt)
            loss = criterion(outputs, lbl)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'comments_classifier.pth')
    # 模型词典
    torch.save(vocab, 'comments_vocab.pth')

   