import torch
import torch.nn as nn
import torch.optim as optim

vocab_size = 100000  # 假设词汇表大小为1000

# 手动实现embedding
class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CustomEmbedding, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_ids):
        return self.embedding_matrix[input_ids]
    
# 定义RNN + FC模型
class TextClassfier(nn.modules):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassfier, self).__init__()
        self.embedding = CustomEmbedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        _, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)  # Remove the sequence dimension
        out = self.fc(hidden)
        return out

# 定义超参数
embeding_dim = 10
hidden_dim = 20
output_dim = 2

# 定义模型
model = TextClassfier(vocab_size, embeding_dim, hidden_dim, output_dim)