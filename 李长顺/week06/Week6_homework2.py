# 2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#    数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2


# 获取数据集，下载下来
# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("smid80/weatherww2")
# print("Path to dataset files:", path)
# path:C:\Users\licha\.cache\kagglehub\datasets\smid80\weatherww2\versions\1

'''加载数据及预处理'''
import torch
import os
import pandas as pd
torch.manual_seed(42)   # 定义随机种子，确保结果可复现

'''检查数据是否存在'''
Data_path = 'Week6/Whether/Summary_of_Weather.csv'

def check_data_exist(Data_path):
    # print(os.getcwd())
    if not os.path.exists(Data_path):
        print("数据不存在")
    else:
        print("找到数据了")

# 加载数据
def load_data(filepath):
    data = pd.read_csv(filepath)
    if 'Date' not in data.columns or 'MaxTemp' not in data.columns:
        raise ValueError("数据集中缺少 'Date' 或 'MaxTemp' 列，请检查数据集格式。")
    # 转换日期格式
    data['Date'] = pd.to_datetime(data['Date'])
    # 按日期排序
    data = data.sort_values('Date')
    # 提取日期和最高气温
    dates = data['Date'].values
    temps = data['MaxTemp'].values.reshape(-1, 1)
    return dates, temps

# 创建时间序列数据集
import numpy as np
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 创建自定义 Dataset
from torch.utils.data import Dataset, DataLoader
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# 创建DataLoader
 # 定义序列长度
def get_DataLoader(X,y,BATCH_SIZE):
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, features)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    train_dataset = WeatherDataset(X_train_tensor, y_train_tensor)
    test_dataset = WeatherDataset(X_test_tensor, y_test_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader,test_loader

# 构建模型
import torch.nn as nn

class WeatherRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播 RNN
        out, _ = self.rnn(x, h0)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
    
# 创建 TensorBoard writer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/weather_prediction')
def train_model(model, optimizer, train_loader, criterion, epochs=20, tag='1day'):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计损失
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # 记录到 TensorBoard
        writer.add_scalar(f'Loss/train_{tag}', epoch_loss, epoch + 1)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    return model

# 由于 `model_5day` 的输出维度为5，需要调整数据生成方式
# 这里简化处理，假设我们可以直接训练预测5天的模型
# 实际上，可能需要更复杂的数据生成策略

# 重新定义 DataLoader 以适应5天预测
# 假设我们希望用前30天的数据预测未来5天的数据
# 需要调整 create_sequences 函数
def create_sequences_5day(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)

def predict_5day(temps_scaled , BATCH_SIZE):
    # 创建5天预测的数据集
    SEQ_LENGTH_5DAY = 30
    PRED_LENGTH_5DAY = 5

    X_5day, y_5day = create_sequences_5day(temps_scaled, SEQ_LENGTH_5DAY, PRED_LENGTH_5DAY)

    # 划分训练集和测试集
    train_size_5day = int(len(X_5day) * 0.8)
    X_train_5day, X_test_5day = X_5day[:train_size_5day], X_5day[train_size_5day:]
    y_train_5day, y_test_5day = y_5day[:train_size_5day], y_5day[train_size_5day:]

    # 转换为 PyTorch 张量
    X_train_tensor_5day = torch.tensor(X_train_5day, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor_5day = torch.tensor(y_train_5day, dtype=torch.float32).unsqueeze(-1)

    X_test_tensor_5day = torch.tensor(X_test_5day, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor_5day = torch.tensor(y_test_5day, dtype=torch.float32).unsqueeze(-1)

    # 创建 Dataset 和 DataLoader
    train_dataset_5day = WeatherDataset(X_train_tensor_5day, y_train_tensor_5day)
    test_dataset_5day = WeatherDataset(X_test_tensor_5day, y_test_tensor_5day)

    train_loader_5day = DataLoader(train_dataset_5day, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_5day = DataLoader(test_dataset_5day, batch_size=BATCH_SIZE, shuffle=False)

    # 调整模型 B 的输出维度为5
    model_5day = WeatherRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE_5, NUM_LAYERS).to(device)

    # 训练模型 B（预测未来5天）
    model_5day_trained = train_model(model_5day, optimizer_5day, train_loader_5day, criterion, epochs=20, tag='5day')
    return test_loader_5day, model_5day_trained

# 定义评估函数
def evaluate_model(model, test_loader, criterion, tag='1day'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    writer.add_scalar(f'Loss/test_{tag}', avg_loss, 20)  # 假设训练20个epoch后评估
    print(f'Test Loss ({tag}): {avg_loss:.4f}')
    return avg_loss

import matplotlib.pyplot as plt

def plot_predictions(model, data_loader, scaler, tag='1day', num_samples=5):
    model.eval()
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(data_loader):
            if i >= num_samples:
                break
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).cpu().numpy()
            batch_X_np = batch_X.squeeze(-1).cpu().numpy()
            batch_y_np = batch_y.squeeze(-1).cpu().numpy()
            
            # 反归一化
            batch_X_np = scaler.inverse_transform(batch_X_np.reshape(-1, 1)).reshape(batch_X_np.shape)
            batch_y_np = scaler.inverse_transform(batch_y_np.reshape(-1, 1)).reshape(batch_y_np.shape)
            outputs = scaler.inverse_transform(outputs.reshape(-1, 1)).reshape(outputs.shape)
            
            # 绘制
            plt.figure(figsize=(10, 4))
            time_steps = np.arange(len(batch_X_np[0])) + 1  # 假设时间步从1开始
            for j in range(len(batch_X_np)):
                plt.plot(time_steps, batch_X_np[j], label=f'Input Day {j+1}' if j == 0 else "", alpha=0.5)
            plt.plot(len(batch_X_np[0]) + np.arange(len(outputs[0])) + 1, outputs[0], 'r-', label='Predicted' if i == 0 else "")
            plt.plot(len(batch_X_np[0]) + np.arange(len(batch_y_np[0])) + 1, batch_y_np[0], 'g--', label='Actual' if i == 0 else "")
            plt.title(f'{tag.capitalize()} Day Prediction Sample {i+1}')
            plt.xlabel('Day')
            plt.ylabel('Max Temperature (Normalized)')
            plt.legend()
            plt.grid(True)
            writer.add_figure(f'Predictions/{tag}_sample_{i+1}', plt.gcf(), global_step=20)
            plt.close()


from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
if __name__ == "__main__":
    # 加载数据了
    date , tem = load_data("Week6/Whether/Summary_of_Weather.csv")
    # 数据归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    temps_scaled = scaler.fit_transform(tem)

    SEQ_LENGTH = 30  # 使用过去30天的数据预测未来
    BATCH_SIZE = 64
    X, y = create_sequences(temps_scaled, SEQ_LENGTH)
    train_loader,test_loader = get_DataLoader(X,y,BATCH_SIZE)
    # print("train_loader:",train_loader)
    # print("test_loader:",test_loader)


    # 实例化模型
    # 定义模型参数
    INPUT_SIZE = 1      # 输入特征维度（最高气温）
    HIDDEN_SIZE = 64    # 隐藏层维度
    OUTPUT_SIZE_1 = 1   # 预测未来1天的输出维度
    OUTPUT_SIZE_5 = 5   # 预测未来5天的输出维度
    NUM_LAYERS = 2      # RNN层数
    model_1day = WeatherRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE_1, NUM_LAYERS)
    model_5day = WeatherRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE_5, NUM_LAYERS)

    # 如果有 GPU，将模型移到 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1day.to(device)
    model_5day.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_1day = optim.Adam(model_1day.parameters(), lr=0.001)
    optimizer_5day = optim.Adam(model_5day.parameters(), lr=0.001)   

    # 训练模型 A（预测未来1天）
    model_1day_trained = train_model(model_1day, optimizer_1day, train_loader, criterion, epochs=20, tag='1day')                                                                               
    test_loader_5day, model_5day_trained = predict_5day(temps_scaled , BATCH_SIZE)

    # 评估模型 A
    evaluate_model(model_1day_trained, test_loader, criterion, tag='1day')

    # 评估模型 B
    evaluate_model(model_5day_trained, test_loader_5day, criterion, tag='5day')

    
    # 可视化模型 A 的预测
    plot_predictions(model_1day_trained, test_loader, scaler, tag='1day')

    # 可视化模型 B 的预测
    plot_predictions(model_5day_trained, test_loader_5day, scaler, tag='5day')

    
