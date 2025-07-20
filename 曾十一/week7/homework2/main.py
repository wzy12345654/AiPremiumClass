import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence  # 长度不同张量填充为相同长度
import jieba
from torch.utils.data import random_split


#词典构建
def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)  # PAD: padding, UNK: unknown
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx


#模型构建
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


#主函数
def mian():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    with open('/mnt/data_1/zfy/4/week7/homework/homework2/comments.pkl','rb') as f:
        comments_data = pickle.load(f)
    
    #构建词汇表
    vocab = build_from_doc(comments_data)

    # 自定义数据转换方法(callback function)回调函数
    def convert_data(batch_data):
        comments, votes = [],[]
        # 分别提取评论和标签
        for comment, vote in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            votes.append(vote)

        # 将评论和标签转换为tensor
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])  # 填充为相同长度
        labels = torch.tensor(votes)
        return commt, labels

    #划分数据集
    train_data, val_data, test_data = random_split(comments_data,[0.7,0.2,0.1])
    batch_size = 64 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=convert_data)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=convert_data)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=convert_data)
    
    #构建模型
    vocab_size = len(vocab)
    embedding_dim = 120
    hidden_size = 128
    num_classes = 2
    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, num_classes)
    model.to(device)

    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

    #模型训练和评估
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs} | Batch Loss: {loss.item():.4f}")

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
        # 计算平均损失和准确率

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_samples += labels.size(0)

        avg_val_loss = val_loss / val_samples
        val_accuracy = val_correct / val_samples
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Acc: {val_accuracy:.4f}")

    # 测试集评估
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    mian()


