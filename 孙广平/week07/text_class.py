import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  


def stratified_split_dataloaders(comments, labels, batch_size=64, test_size=0.2, shuffle=True, seed=42):
    """
    按照标签进行分层划分，并返回训练和验证 DataLoader。

    参数:
        comments (Tensor): 输入评论张量 (N, L)
        labels (Tensor): 标签张量 (N,)
        batch_size (int): DataLoader 的批大小
        test_size (float): 验证集占比
        shuffle (bool): 是否打乱训练集
        seed (int): 随机种子

    返回:
        train_loader, val_loader: 分层后的 DataLoader 对象
    """
    comments_np = comments.cpu().numpy()
    labels_np = labels.cpu().numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        comments_np, labels_np,
        test_size=test_size,
        random_state=seed,
        stratify=labels_np
    )

    # 转回 Tensor
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 封装为 Dataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)



def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples



class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        output, _ = self.lstm(embedded)
        weights = torch.softmax(self.attn(output), dim=1)  
        attn_output = torch.sum(weights * output, dim=1)  
        output = self.fc(self.dropout(attn_output))
        return output



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备
    comments = torch.load('./data/comments_jieba.pkl')
    labels = torch.load('./data/labels_jieba.pkl')
    
    dataset = TensorDataset(comments, labels)

    # 划分训练集和验证集
    train_loader, val_loader = stratified_split_dataloaders(comments, labels, batch_size=64, test_size=0.2)
    
    # 模型参数
    vocab_size = 276218  # 词汇表大小
    embedding_dim = 100  # 嵌入维度
    hidden_size = 128  # LSTM 隐藏层大小
    num_classes = 2  # 分类数量
    num_epochs = 10  # 训练轮数
    learning_rate = 0.001  # 学习率

    # 初始化模型
    model = Comments_Classifier(vocab_size, embedding_dim, hidden_size, num_classes).to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir='./logs')  

    # 训练模型
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)

        # 将指标写入 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 关闭 TensorBoard
    writer.close()

    # 保存模型
    torch.save(model.state_dict(), './model/comments_classifier.pth')