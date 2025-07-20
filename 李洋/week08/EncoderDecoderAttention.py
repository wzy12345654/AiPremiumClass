# -*- coding: utf-8 -*-
# @Date    : 2025/4/28 16:25
# @Author  : Lee
import torch
import torch.nn as nn
import pickle


class Encoder(nn.Module):
    def __init__(self,input_dim,emb_size,hidden_size,dropout=0.5):
        super(Encoder,self).__init__()
        self.emb = nn.Embedding(input_dim,emb_size)
        self.rnn = nn.GRU(emb_size,hidden_size,batch_first=True,bidirectional=True,dropout=dropout)

    def forward(self,x):
        x = self.emb(x)
        out,hidden_e = self.rnn(x)
        return torch.cat((hidden_e[0],hidden_e[1]),dim=-1),out

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()

    def forward(self,encoder,decoder):
        a_t = torch.bmm(encoder,decoder.permute(0,2,1))
        sa_t = torch.softmax(a_t,dim=1)
        c_t = torch.bmm(sa_t.permute(0,2,1),encoder)
        return c_t

class Decoder(nn.Module):
    def __init__(self,input_dim,emb_size,hidden_size,dropout=0.5):
        super(Decoder,self).__init__()
        self.emb = nn.Embedding(input_dim,emb_size)
        self.rnn = nn.GRU(emb_size,hidden_size*2,batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size*2,input_dim)
        self.attention = Attention()
        self.attention_fc = nn.Linear(hidden_size*4,hidden_size*2)

    def forward(self,token,hidden_e,out_e):
        token = self.emb(token)
        out_d,hidden_d = self.rnn(token,hidden_e.unsqueeze(0))
        c_t = self.attention(out_e,out_d)
        cat_c_t = torch.cat((out_d,c_t),dim=-1)
        out_d = torch.tanh(self.attention_fc(cat_c_t))
        out_d = self.fc(out_d)
        return out_d,hidden_d

class Sqe2seq(nn.Module):
    def __init__(self,input_e,input_d,emb_size,hidden_size,dropout=0.5):
        super(Sqe2seq,self).__init__()
        self.encoder = Encoder(input_e,emb_size,hidden_size,dropout=dropout)
        self.decoder = Decoder(input_d,emb_size,hidden_size,dropout=dropout)

    def forward(self,e,d):
        hidden_e,out_e = self.encoder(e)
        out_d,hidden_d = self.decoder(d,hidden_e,out_e)
        return out_d,hidden_d


