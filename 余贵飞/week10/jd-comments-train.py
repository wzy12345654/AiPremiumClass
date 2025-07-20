
import pandas as pd
import jdDataProcess as jd_comments_process
import commentsClassification as jd_comments_classification
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

def train_model(dataProcess=True, epochs=3, lr=2e-5):
    # 1. 数据预处理类
    if dataProcess:
        # 如果需要预处理数据，则调用 jd_comments_process 中的 dataProcess 函数
        jd_comments_process.dataProcess()

    # 2. 读取处理后的文件数据
    df = pd.read_csv("/kaggle/working/jd-comments.csv")
    print(df[0:5])
    # 划分训练集和测试集
    train_contents, test_contents, train_labels, test_labels = train_test_split(
        df['content'], df['label'], test_size=0.2, random_state=42
    )
    
    # 加载预训练的BERT分词器
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载预训练的BERT模型
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 创建训练集和测试集的数据加载器
    train_dataset = jd_comments_classification.SentimentDataset(train_contents, train_labels, tokenizer)
    val_dataset = jd_comments_classification.SentimentDataset(test_contents, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    

    # 训练模型
    jd_comments_classification.train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=epochs, lr=lr)