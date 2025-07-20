import csv
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# rnn模型
# rnn模型
class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        # 修改全连接层的输出维度为 5
        self.fc = torch.nn.Linear(64, 5)

    def forward(self, x):
        x, h = self.rnn(x)
        # 取最后一个时间步的隐藏状态
        x = x[:, -1, :]
        y = self.fc(x)
        # 添加一个维度，使其形状为 (batch_size, 5, 1)
        y = y.unsqueeze(-1)
        return y


# 绘制图像 y_pred为预测值，y为实际值
def plot_series(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$"):
    # 设置子图的数量为3行5列
    r, c = 3, 5
    # sharey=True 和 sharex=True 表示所有子图共享 y 轴和 x 轴的刻度。figsize=(20, 10)
    # fig表示画框，可以控制图像大小、分辨率等，axes是每一个画框里的画布
    fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))
    for row in range(r):
        for col in range(c):
            # 设置当前的活动子图为 axes[row][col]
            plt.sca(axes[row][col])
            ix = col + row * c
            # 取出某一行的数据，用点表示数据位置，用实线进行连接
            plt.plot(series[ix, :], ".-")
            # 绘制目标值，起始位置x坐标为len(series[ix, :])到len(series[ix, :])+len(y[ix])，y坐标为y[ix]，bx表示蓝色的叉
            if y is not None:
                for j in range(y.shape[1]):
                    plt.plot(range(len(series[ix, :]) + j, len(series[ix, :]) + j + 1), y[ix, j], "bx", markersize=10)
            # 绘制预测值，圆的红点
            if y_pred is not None:
                for j in range(y_pred.shape[1]):
                    plt.plot(range(len(series[ix, :]) + j, len(series[ix, :]) + j + 1), y_pred[ix, j], "ro")
            # 绘制预测值加上和减去标准差的曲线
            if y_pred_std is not None:
                for j in range(y_pred.shape[1]):
                    plt.plot(range(len(series[ix, :]) + j, len(series[ix, :]) + j + 1), y_pred[ix, j] + y_pred_std[ix, j])
                    plt.plot(range(len(series[ix, :]) + j, len(series[ix, :]) + j + 1), y_pred[ix, j] - y_pred_std[ix, j])
            # 是否开启网格
            plt.grid(True)
            # plt.hlines(0, 0, 100, linewidth=1)
            # plt.axis([0, len(series[ix, :])+len(y[ix]), -1, 1])
            if x_label and row == r - 1:
                plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
                plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.show()


def generate_time_series(temp_values, batch_size, n_steps):
    # 生成用于填充到数据集矩阵，一共 batch_size 条数据，取其中 n_steps - 5 条预测后 5 条
    series = np.zeros((batch_size, n_steps))  # (batch_size, n_steps)
    # 生成随机样本索引，sta_size 表示有多少个气象站
    sta_size = len(temp_values)
    # 从 0 到 sta_size 随机生成 batch_size 条数据
    sta_idx = np.random.randint(0, sta_size, batch_size)  # (batch_size,)
    print(sta_idx)

    for i, idx in enumerate(sta_idx):
        # temp_values 结构 [[],[],[]]
        temps = temp_values[idx]  # 随机获取某个气象站的温度数据
        # 判断这个气象站数据量有多少
        temp_size = len(temps)
        # 随机选取一个位置，获取这个位置之后的 n_steps 条数据
        rnd_idx = np.random.randint(0, temp_size - n_steps)
        # 每一行都是一个气象站
        series[i] = np.array(temps[rnd_idx:rnd_idx + n_steps])  # series (batch_size, n_steps)
    # 返回 X 和 y  X(batch_size, n_steps - 5, 1) y (batch_size, 5, 1)
    # 这里 X 取为 batch_size 个气象站 n_steps - 5 天的数据，y 取最后 5 天的数据
    return series[:, :n_steps - 5, np.newaxis].astype(np.float32), series[:, -5:, np.newaxis].astype(np.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        if self.train:
            return torch.from_numpy(self.X[ix]), torch.from_numpy(self.y[ix])
        return torch.from_numpy(self.X[ix])


if __name__ == "__main__":
    writer = SummaryWriter()
    # 第一步：获取数据，并处理脏数据
    # 所有气象站最高气温数据,最后结构为一个键对一个数组，数组中是温度数据
    stations_maxtemp = {}
    with open('week6/Summary of Weather.csv') as f:
        reader = csv.DictReader(f)
        for item in reader:
            sta = item['STA']
            stations_maxtemp[sta] = stations_maxtemp.get(sta, [])
            stations_maxtemp[sta].append(float(item['MaxTemp']))

    # 只绘制站点存有数据，且数据量大于20的站点天气曲线,此时max_temps结构为[[],[],[]]
    max_temps = [temps for temps in stations_maxtemp.values() if len(temps) > 20]

    # 过滤掉温度小于-17的极寒异常值,filted_maxtemps结构也为[[],[],[]]
    filted_maxtemps = [[temp for temp in temps if temp > -17] for temps in max_temps]

    # 第二步 将数据拆分为训练与测试，根据题目要求：通过连续的20组数据，预测第21组数据的结果
    n_steps = 25
    max_temps = filted_maxtemps
    X_train, y_train = generate_time_series(max_temps, 7000, n_steps)
    X_test, y_test = generate_time_series(max_temps, 2000, n_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 转换为 PyTorch 张量
    train_data = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_labels = torch.tensor(y_train, dtype=torch.float32).to(device)
    test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 创建数据加载器,先进行封装以便一一对应
    train_dataset = TimeSeriesDataset(train_data.cpu().numpy(), train_labels.cpu().numpy())
    test_dataset = TimeSeriesDataset(test_data.cpu().numpy(), test_labels.cpu().numpy())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 实例化模型
    model = RNN()
    model.to(device)
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 200
    # tqdm进度条函数
    bar = tqdm(range(1, num_epochs + 1))
    for epoch in bar:
        model.train()
        train_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # 这里移除了squeeze操作
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # 评估模型 model.eval() 是为了让模型结构在评估时表现正确；而 with torch.no_grad() 是为了提高效率并避免不必要的梯度计算。两者配合使用才是最佳实践。
        model.eval()
        eval_loss = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                y_hat = model(images)
                loss = criterion(y_hat, labels)
                eval_loss.append(loss.item())
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")

    # 进行预测
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # 打印一些预测结果示例（可选）
    # print("Predictions vs True Labels (Sample):")
    # for pred, true in zip(all_predictions[:10], all_true_labels[:10]):
    #     print(f"Prediction: {pred}, True Label: {true}")

    # 提取用于绘制的 series 数据，去掉最后一个维度
    series = X_test.squeeze(axis=-1)

    # 调用 plot_series 函数绘制预测结果
    plot_series(series, y=all_true_labels.reshape(-1, 5), y_pred=all_predictions.reshape(-1, 5))