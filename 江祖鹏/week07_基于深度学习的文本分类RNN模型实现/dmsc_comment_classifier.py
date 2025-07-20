import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import pickle

def build_from_doc(doc):
    # 用set集合去重
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    #set集合转换成列表,固定顺序（set是无序的）
    vocab = ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown\
    #把评论转化为索引(用enumerate函数给每个单词分配一个索引)
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

class Comments_Classifier(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_classes): 
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        # output: (batch_size, seq_len, hidden_size)
        output, (hidden, _) = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        
        return output
    

if __name__ == '__main__':
    #设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #加载训练语料
    with open('ds_comments.pkl','rb') as f:
        comments_data = pickle.load(f)

    #构建词汇表
    vocab = build_from_doc(comments_data)
    print('词汇表大小:', len(vocab))

    #所有向量集合 Embedding（词嵌入）
    emb = nn.Embedding(len(vocab),100) # 词汇表大小，向量维度

    #回调函数，自定义数据转换方法
    #该函数会在每个batch数据加载时被调用
    def convert_data(batch_data):
        comments, votes = [],[]
        #分别提取评论和标签
        for comment,vote in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK'])for word in comment]))
            votes.append(vote)

        #将评论和标签转换为tensor
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])
        lables = torch.tensor(votes)
        return commt, lables
    #通过Dataset构建DataLoader
    dataloader = DataLoader(comments_data, batch_size=32, shuffle=True, collate_fn=convert_data)


    #模型参数
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2

    #构建模型
    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, num_classes).to(device)

    #损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #训练模型
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (comm,lable) in enumerate(dataloader):
            #将数据移动到设备上
            comm = comm.to(device)
            lable = lable.to(device)

            #前向传播
            outputs = model(comm)
            #计算损失
            loss = criterion(outputs, lable)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            #更新参数
            optimizer.step()
            if (i+1) % 200 == 0:
                print(f'Epoch[{epoch}/{num_epochs}],step[{i}/{len(dataloader)}],loss:{loss.item():.4f}')

    #保存模型
    torch.save(model.state_dict(),'dmsc_comments_classifier.pth')
    #保存词典
    torch.save(vocab,'dmsc_vocab.pth')
            