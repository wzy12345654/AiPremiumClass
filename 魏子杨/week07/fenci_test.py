import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba


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
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding_dim = 100
hidden_size = 128
num_classes = 2

# 加载词典
vocab = torch.load('comments_vocab.pth')
# 测试模型
comment1 = '这部电影真好看！全程无尿点'
comment2 = '看到一半就不想看了，太无聊了，演员演技也很差'

# 将评论转换为索引
comment1_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment1)])
comment2_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment2)])
# 将评论转换为tensor
comment1_idx = comment1_idx.unsqueeze(0).to(device)  # 添加batch维度    
comment2_idx = comment2_idx.unsqueeze(0).to(device)  # 添加batch维度

# 加载模型
model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
model.load_state_dict(torch.load('comments_classifier.pth'))
model.to(device)

# 模型推理
pred1 = model(comment1_idx)
pred2 = model(comment2_idx)

# 取最大值的索引作为预测结果
pred1 = torch.argmax(pred1, dim=1).item()
pred2 = torch.argmax(pred2, dim=1).item()
print(f'评论1预测结果: {pred1}')
print(f'评论2预测结果: {pred2}')