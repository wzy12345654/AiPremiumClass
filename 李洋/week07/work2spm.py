import sentencepiece  as smp
import pandas as pd
import csv
import pickle
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

df = pd.read_csv('DMSC.CSV')

with open('dmc.txt', 'w',encoding='utf-8') as file:
    for lines in df["Comment"]:
        line = lines + '\n'
        file.write(line)
#
#
smp.SentencePieceTrainer.Train(input='dmc.txt', model_prefix='dmc_model',vocab_size=100000,pad_id=0,unk_id=3,user_defined_symbols=['<pad>','unk'])

sp = smp.SentencePieceProcessor(model_file='dmc_model.model')

ds = []
with open('DMSC.csv','r',encoding='utf-8') as file1:
    reader = csv.DictReader(file1)
    for row in reader:
        label = int(row['Star'])
        if label  in [0,1,2,4,5]:
            work = sp.EncodeAsPieces(row['Comment'])
            vocab = 1 if label in [0,1,2] else 0
            ds.append((work,vocab))

with open('DMSC.pkl','wb') as file2:
    pickle.dump(ds,file2)

with open('DMSC.pkl','rb') as file3:
    comment_data = pickle.load(file3)



#模型
class Emb1(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,num_factory,layers):
        super(Emb1,self).__init__()
        self.emb = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size,hidden_size,num_layers=layers,batch_first=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size,num_factory)

    def forward(self,x):
        x = self.emb(x)
        output,_ =self.lstm(x)
        out = output[:,-1,:]
        out = self.relu(out)
        out = self.drop(out)
        output = self.fc(out)
        return output

def coll_classes(vocab):
    def coll_fn(datas):
        comments,votes = [],[]
        for comment,vote in datas:
            comments.append(torch.tensor([vocab.get(work,vocab['<unk>']) for work in comment]))
            votes.append(vote)
        comm = pad_sequence(comments,batch_first=True,padding_value=vocab['<pad>'])
        label = torch.tensor(votes)
        return comm,label
    return coll_fn

if __name__ == '__main__':

    device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('vocab.pkl', 'rb') as f:
        comment_data = pickle.load(f)

    vocab = {sp.IdToPiece(line):line for line in range(len(sp))}

    train_datas,test_datas = train_test_split(comment_data,test_size=0.4)

    coll_class = coll_classes(vocab)

    train_dataloader = DataLoader(train_datas,batch_size=64,shuffle=True,collate_fn=coll_class)
    test_dataloader =  DataLoader(test_datas,batch_size=64,shuffle=False,collate_fn=coll_class)
    # 超参数
    epochs = 20
    voca_size = len(vocab)
    embedding = 200
    hidden_size = 150
    num_factory = 2
    layers = 2

    #优化器和损失
    model = Emb1(voca_size,embedding,hidden_size,num_factory,layers).to(device)
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

    torch.save(model.state_dict(),'spm_dmc_model.pth')
    torch.save(vocab,'spm_dmc_vocab.pth')