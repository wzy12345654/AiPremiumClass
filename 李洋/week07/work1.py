from itertools import count

import jieba
import pandas as pd
import pickle
import csv

ds_dict = []
count = 0
with open('DMSC.csv','r',encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if count > 300000:
            break
        label = int(row['Star'])
        if label in [0,1,2,4,5]:
            work = jieba.lcut(row['Comment'])
            vocab = 1 if label in [0,1,2] else 0
            ds_dict.append((work,vocab))
        count+=1

with open('vocab.pkl','wb') as f1:
    pickle.dump(ds_dict,f1)

