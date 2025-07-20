import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 创建时间序列数据集
def create_time_series_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps, 0])  # 假设目标是第一列，即MaxTemp
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 加载数据时使用多个工作线程
def load_data(file_path, time_steps, batch_size, num_workers):
    data = pd.read_csv(file_path)
    data = data.values
    X, y = create_time_series_dataset(data, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 使用最后一个时间步的输出
        return out

# 预测函数
def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.view(-1, 50, 6).to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)

# 主函数
def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    num_workers = 8  # 使用多个工作线程
    time_steps = 50
    batch_size = 128

    # 加载数据
    train_loader, test_loader = load_data('/mnt/data_1/zfy/4/week6/资料/homework_2/processed_weather_data.csv', time_steps=time_steps, batch_size=batch_size, num_workers=num_workers)
    
    input_size = 6  # 输入特征的数量
    hidden_size = 64  # 隐藏层的大小
    output_size = 1  # 输出特征的数量

    # 创建模型实例
    model = RNN(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.view(-1, time_steps, input_size).to(device)
            labels = labels.view(-1, 1).to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每500步记录一次训练损失
            if (i + 1) % 500 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)

        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for inputs, labels in test_loader:
                inputs = inputs.view(-1, time_steps, input_size).to(device)
                labels = labels.view(-1, 1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

            test_loss /= len(test_loader)
            print(f'Test Loss: {test_loss:.4f}')
            writer.add_scalar('test loss', test_loss, epoch)

            
        # 记录每个epoch的预测结果与真实值的对比
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(-1, time_steps, input_size).to(device)
                labels = labels.view(-1, 1).to(device)

                outputs = model(inputs)
                y_true.append(labels.cpu())
                y_pred.append(outputs.cpu())

        # 将预测结果转换为numpy数组
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        # 可视化预测结果与真实值的比较
        # 用 add_scalars 记录每个真实值和预测值
        for i in range(100):
            writer.add_scalars('prediction vs real', 
                               {'real': y_true[i], 'predicted': y_pred[i]}, global_step=epoch)

        # 记录预测结果的直方图
        writer.add_histogram('predictions', y_pred, epoch)


        # 记录模型的计算图（仅记录第一次）
        if epoch == 0:
            writer.add_graph(model, inputs)

    writer.close()

# 运行主函数
if __name__ == '__main__':
    main()

# 启动 TensorBoard
# 在命令行中运行：tensorboard --logdir=runs
