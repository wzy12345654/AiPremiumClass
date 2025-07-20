import data_douban as dataload
from collections import Counter
import torch
import torch.nn as nn
import json
import fetch_vocab as Vocabulary
import fetch_LSTMModel as LSTMModel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 加载数据
    df = dataload.load_data()
    # 数据过滤
    df = df[df['Star'].isin([1, 2, 4, 5])]
    # 新增一列为标签,处理为 negetive 或者 positive
    df ['label'] = df["Star"].apply(dataload.fetch_star2cate)
    # 新增一列为分词结果
    df ['tokens'] = df ['Comment'].apply(dataload.fetch_text_wcut)
    all_cut_comment = []
    for words in df['tokens']:
        all_cut_comment.extend(words)

    word_count = Counter(all_cut_comment)
    # 按词频排序
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    # 保留前10000个高频词
    vocab = [word for word, _ in sorted_words[:10000]]
    # 添加特殊词
    vocab = ['<PAD>', '<UNK>'] + vocab
    # 构建词到索引的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # 准备训练数据，df['tokens'],df['label']
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = LSTMModel.DoubanDataset(train_data, word_to_idx)
    test_dataset = LSTMModel.DoubanDataset(test_data, word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

    # 定义模型参数
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = LSTMModel.LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 5
    LSTMModel.train_model(model, train_loader, criterion, optimizer, device, epochs)

    # 评估模型
    LSTMModel.evaluate_model(model, test_loader, device)
