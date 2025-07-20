import kagglehub

# Download latest version
##path = kagglehub.dataset_download("utmhikari/doubanmovieshortcomments")

#print("Path to dataset files:", path)

import jieba
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

#第一步：数据加载与预处理
ds_comments = []
#with open('DMSC.csv', 'r', newline='', encoding='utf-8') as f:
#    reader = csv.DictReader(f)
#    for row in reader:
#        #print(row['Comment'], row['Star'])#[0-5]
#        if row['Star'] == None or row['Star'] == 3: continue
#        label = 1 
#        if int(row['Star']) > 3: label = 0
#        ds_comments.append((row['Comment'], label)) # 0-1标签

#comments_len = [len(c) for c,v in ds_comments]
#plt.hist(comments_len, bins=50)
#plt.xlabel('Length of comments')
#plt.ylabel('Number of comments')
#plt.title('Distribution of comments length')
#plt.show()
#plt.boxplot(comments_len) # 绘制箱线图
#ds_comments = [values for values in ds_comments if len(values[0]) in range(5,150)] # 过滤掉空评论
#with open('comments.pkl', 'wb') as f:
#    pickle.dump(ds_comments, f)
with open('comments.pkl', 'rb') as f:
    ds_comments = pickle.load(f)


#第二步:构建词汇表
#def from_text(doc):
#        vocab = set()
#        for line in doc:    
#            words = jieba.lcut(line[0])
#            vocab.update(words)

#        vocab = ['PAD','UNK'] + list(vocab)  
#        word2idx = {word: i for i, word in enumerate(vocab)}
#        return vocab, word2idx  
#vocabulary, word2idx= from_text(ds_comments) # 词汇表
#torch.save(vocabulary, 'vocabulary.pkl') # 保存词汇表

#加载词汇表
vocabulary = torch.load('vocabulary.pkl')
word2idx = {word: i for i, word in enumerate(vocabulary)}


#第三步：构建数据集
from torch.nn.utils.rnn import pad_sequence #填充序列
#自定义数据转换方法（callback function,回调函数）
def convert_data(batch_data):
    textx,votes = [],[]
    #分别提取评论和标签
    for comment,vote in batch_data:
        textx.append(torch.tensor([word2idx.get(word, 1) for word in jieba.lcut(comment)])) # 如果word不在字典中，则返回0，即UNK的位置，对于OOV问题的解决方案
        votes.append(vote)
    votes = torch.tensor(votes)
    #对评论进行padding
    textx = pad_sequence(textx, batch_first=True, padding_value=0) # 填充序列
    return textx, votes
from torch.utils.data import Dataset, DataLoader
dataLoader = DataLoader(ds_comments, batch_size=128, shuffle=True,collate_fn=convert_data) # 数据加载器,在进行数据读取的时候自定义函数会被启用，惰性方法

class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(Comments_Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # padding_idx=0表示填充的索引
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])#取最后一个时间步的输出交给全连接层
        return x

# 构建模型
vocab_size = len(vocabulary)
embedding_dim = 100
hidden_dim = 128
output_dim = 2 # 二分类
# 这里的output_dim可以根据需要进行修改
model = Comments_Classifier(vocab_size, embedding_dim, hidden_dim, output_dim) # 2分类
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam优化器
# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    for i, (textx, votes) in enumerate(dataLoader):
        outputs = model(textx)
        loss = criterion(outputs, votes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataLoader)}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'comments_classifier.pth')

#加载模型
model = Comments_Classifier(len(vocabulary), embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('comments_classifier.pth'))    