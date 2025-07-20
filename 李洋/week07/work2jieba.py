# -*- coding: utf-8 -*-
# @Date    : 2025/4/14 14:45
# @Author  : Lee
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])
    vocab = ['PAD','UNK'] +list(vocab)
    w2id = {work:idx for idx,work in enumerate(vocab)}
    return w2id

#模型
class Emb(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,num_factory):
        super(Emb,self).__init__()
        self.emb = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_factory)

    def forward(self,x):
        x = self.emb(x)
        output,_ =self.lstm(x)
        output = self.fc(output[:,-1,:])
        return output

def coll_classes(vocab):
    def coll_fn(datas):
        comments,votes = [],[]
        for comment,vote in datas:
            comments.append(torch.tensor([vocab.get(work,vocab['UNK']) for work in comment]))
            votes.append(vote)
        comm = pad_sequence(comments,batch_first=True,padding_value=vocab['PAD'])
        label = torch.tensor(votes)
        return comm,label
    return coll_fn

if __name__ == '__main__':

    device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('vocab.pkl', 'rb') as f:
        comment_data = pickle.load(f)

    vocab = build_from_doc(comment_data)

    train_datas,test_datas = train_test_split(comment_data,test_size=0.4)

    coll_class = coll_classes(vocab)

    train_dataloader = DataLoader(train_datas,batch_size=64,shuffle=True,collate_fn=coll_class)
    test_dataloader =  DataLoader(test_datas,batch_size=64,shuffle=False,collate_fn=coll_class)
    # 超参数
    epochs = 10
    voca_size = len(vocab)
    embedding = 200
    hidden_size = 150
    num_factory = 2

    #优化器和损失
    model = Emb(voca_size,embedding,hidden_size,num_factory).to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

    #模型训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_num = 0
        for comment,target in train_dataloader:
            comment,target = comment.to(device),target.to(device)
            output = model(comment)
            loss_val = loss(output,target)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            train_loss += loss_val.item()
            train_num += 1
        train_avg_loss = train_loss/train_num
        print(f'{epoch}/{epochs},train_avg_loss:{train_avg_loss}')

        model.eval()
        test_loss = 0
        test_num = 0
        with torch.no_grad():
            for comment,target in test_dataloader:
                comment, target = comment.to(device), target.to(device)
                output = model(comment)
                loss_val = loss(output,target)
                test_loss += loss_val.item()
                test_num += 1
        test_avg_loss = test_loss/test_num
        print(f'{epoch}/{epochs},test_avg_loss:{test_avg_loss}')


    torch.save(model.state_dict(),'coment_emb.pth')

    torch.save(vocab,'commrnt_vocab.pth')

