import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def build_from_doc(docs):
    vocab = set()
    for line in docs:
        vocab.update(line[0])
    vocab = ['PAD', 'UNK'] + list(vocab)  # PAD:padding, UNK: unknown
    # 构建词汇到索引的映射
    w2idx = {word: i for i, word in enumerate(vocab)}
    return w2idx

class Comments_Classfier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    with open ('data/db_comment.pkl', 'rb') as f:
        db_comment = pickle.load(f)

    # 构建词汇表
    vocab = build_from_doc(db_comment)

    # 索引 -> 向量
    # 所有向量的集合 词嵌入embedding
    emb = nn.Embedding(len(vocab), 100)  # 假设词嵌入维度为100

    # 将文本转换为索引序列
    # db_comment_idx = []
    # for words, label in db_comment:
    #     idx_seq = [vocab.get(word, vocab['UNK']) for word in words]
    #     db_comment_idx.append(torch.tensor(idx_seq, dtype=torch.long))

    # 自定义回调函数
    def convert_data(batch_data):
        comments, stars = [], []
        for comment, star in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            stars.append(star)
        # 填充序列
        comments = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])
        labels = torch.tensor(stars, dtype=torch.long)
        return comments, labels

    dataloader = DataLoader(db_comment, batch_size=32, shuffle=True, collate_fn=convert_data)

    # 构建模型
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # 二分类

    model = Comments_Classfier(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            comments, labels = batch
            comments = comments.to(device)
            labels = labels.to(device)
            # 前向传播
            optimizer.zero_grad()
            outputs = model(comments)
            # 计算损失,反向传播和优化
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if  i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'model/text_classifier.pth')
    # 保存词汇表
    torch.save(vocab, 'model/comments_vocab.pth')