import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
# 加载数据并返回 DataLoader
def load_data(data_home, batch_size):
    # 加载数据集
    dataset = fetch_olivetti_faces(data_home=data_home, shuffle=True, random_state=42)

    # 提取数据和标签
    data, target = dataset.data, dataset.target

    # 将数据集拆分为训练集和测试集
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # 将数据转换为 PyTorch 张量
    data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
    data_test_tensor = torch.tensor(data_test, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train, dtype=torch.long)
    target_test_tensor = torch.tensor(target_test, dtype=torch.long)

    # 使用 TensorDataset 来包装数据和标签
    train_dataset = TensorDataset(data_train_tensor, target_train_tensor)
    test_dataset = TensorDataset(data_test_tensor, target_test_tensor)
    
    # 创建 DataLoader 用于批处理
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #print(f"Training data shape: {data_train_tensor.shape}")
    #print(f"Testing data shape: {data_test_tensor.shape}")
    #print(f"Training labels shape: {target_train_tensor.shape}")
    #print(f"Testing labels shape: {target_test_tensor.shape}")
    
    return train_loader, test_loader






# 定义RNN模型

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out



class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

  


#定义损失函数
def loss_function():
    # 定义交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


#定义优化器
def optimizer(model, learning_rate):
    # 定义Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer



#主函数
# 在训练过程中修改打印和 TensorBoard 写入

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    criterion = loss_function()

    # 创建 TensorBoard SummaryWriter
    writer = SummaryWriter()

    # 训练模型
    num_epochs = 300
    batch_size = 64
    input_size = 4096  # 每个图像的特征数
    hidden_size = 128  # 隐藏层神经元数量
    output_size = 40   # 类别数
    model_RNN = RNN(input_size, hidden_size, output_size)
    model_LSTM = LSTM(input_size, hidden_size, output_size)
    optimizer_RNN = optimizer(model_RNN, learning_rate)
    optimizer_LSTM = optimizer(model_LSTM, learning_rate)

    # 加载数据集  
    data_home = '/mnt/data_1/zfy/4/week6/资料/homework_1'
    train_data_loader, test_data_loader = load_data(data_home=data_home, batch_size=batch_size)

    # 训练RNN模型
    model_RNN.to(device)
    for epoch in range(num_epochs):
        model_RNN.train()
        for i, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer_RNN.zero_grad()
            outputs = model_RNN(images.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_RNN.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss', loss.item(), epoch * len(train_data_loader) + i)
        # 评估RNN模型
        model_RNN.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_RNN(images.unsqueeze(1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_RNN = 100 * correct / total
            print(f'Accuracy of the RNN model on the test images: {accuracy_RNN:.2f}%')
            writer.add_scalar('Accuracy/Train', accuracy_RNN, epoch)
    # 训练LSTM模型
    model_LSTM.to(device)
    for epoch in range(num_epochs):
        model_LSTM.train()
        for i, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer_LSTM.zero_grad()
            outputs = model_LSTM(images.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_LSTM.step()
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss', loss.item(), epoch * len(train_data_loader) + i)

        model_LSTM.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_LSTM(images.unsqueeze(1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_LSTM = 100 * correct / total
            print(f'Accuracy of the LSTM model on the test images: {accuracy_LSTM:.2f}%')
            writer.add_scalar('Accuracy/Train', accuracy_LSTM, epoch)

    # 关闭 TensorBoard SummaryWriter
    writer.close()

if __name__ == '__main__':
    main()
# 运行 TensorBoard
# tensorboard --logdir=/mnt/data_1/zfy/4/week6/资料/homework_1/runs/Apr09_23-22-53_hg-Super-Server
