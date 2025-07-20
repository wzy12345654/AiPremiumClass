import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
    

# 训练模型
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().squeeze()
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy}')

# 自定义数据集类
class DoubanDataset(Dataset):
    def __init__(self, data, word_to_idx, max_length=50):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx]['tokens']
        label = self.data.iloc[idx]['label']
        # 将词语转换为索引
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in tokens]
        # 填充或截断
        if len(indices) < self.max_length:
            indices += [self.word_to_idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        return torch.tensor(indices), torch.tensor(label)