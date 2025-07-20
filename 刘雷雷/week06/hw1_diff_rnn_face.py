# 作业1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，
# 如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html
import os
import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))


class RNN_Classifier(nn.Module):

    def __init__(self, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type
        self.__init_rnn__(rnn_type)
        self.fc = nn.Linear(128, 40)  # 输出层

    def __init_rnn__(self, rnn_type="LSTM"):
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=64,
                hidden_size=128,
                bias=True,
                num_layers=2,
                batch_first=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=64,
                hidden_size=128,
                bias=True,
                num_layers=2,
                batch_first=True,
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=64,
                hidden_size=128,
                bias=True,
                num_layers=2,
                batch_first=True,
            )
        elif rnn_type == "BiRNN":
            self.rnn = nn.RNN(
                input_size=64,
                hidden_size=128,
                bias=True,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
        elif rnn_type == "BiLSTM":
            self.rnn = nn.LSTM(
                input_size=64,
                hidden_size=128,
                bias=True,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
        else:
            raise ValueError("Unsupported RNN type: {}".format(rnn_type))

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        if rnn_type in ["BiRNN", "BiLSTM"]:
            outputs = outputs[:, :, :128] + outputs[:, :, 128:]  # 双向RNN
        out = self.fc(outputs[:, -1, :])
        return out


if __name__ == "__main__":

    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    olivetti_faces = fetch_olivetti_faces(data_home=f"{current_dir}/face_data")

    # 2. Preprocess dataset
    data = olivetti_faces.images
    data = torch.tensor(data, dtype=torch.float32)

    labels = olivetti_faces.target
    labels = torch.tensor(labels, dtype=torch.long)

    # 3. 将数据分为测试和训练集
    idx = torch.randperm(data.size(0))
    data = data[idx]
    labels = labels[idx]

    train_data = data[:300]
    train_labels = labels[:300]
    test_data = data[300:]
    test_labels = labels[300:]

    # dataloader
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # 4. Initialize model
    def build_rnn_model(rnn_type="LSTM"):
        model = RNN_Classifier(rnn_type=rnn_type).to(device)
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, criterion, optimizer

    rnn_types = ["BiRNN", "BiLSTM", "LSTM", "GRU", "RNN"]

    for rnn_type in rnn_types:
        # 模型
        model, criterion, optimizer = build_rnn_model(rnn_type=rnn_type)
        # 5. Train model
        num_epochs = 200
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images.squeeze())
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], {rnn_type} Loss: {loss.item():.4f}"
                    )
                    writer.add_scalar(
                        f"{rnn_type} training loss",
                        loss.item(),
                        epoch * len(train_loader) + i,
                    )
            # 模型评估
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images.squeeze())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], {rnn_type} Test Accuracy: {accuracy:.2f}%"
                )
                writer.add_scalar(f"{rnn_type} test accuracy", accuracy, epoch)
