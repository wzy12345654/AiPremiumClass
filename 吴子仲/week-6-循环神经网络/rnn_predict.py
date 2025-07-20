import torch.nn as nn
import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rnn_model_cls import RNN_Classfier
from gru_model_cls import GRU_Classfier
from lstm_model_cls import LSTM_Classfier

olt_faces = fetch_olivetti_faces(data_home='face_data', shuffle=True)

# 定义超参数
LR = 1e-3
BATCH_SIZE = 10
epochs = 10

# 训练数据
def training_data(epochs, model, train_dl):
    loss_his = []
    # 开启训练模式
    model.train()
    for epoch in range(epochs):
        for i,(img,lbl) in enumerate(train_dl):
            img,lbl = img.to(device), lbl.to(device)
            result = model(img.squeeze())       # LSTM输出需要修改
            loss = loss_fn(result.squeeze(), lbl)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_his.append(loss.item())
            if i % 100 == 0:
                print(f'epoch: {epoch}, loss: {loss.item():.4f}')
                writer.add_scalar('trainning loss', loss.item(), epoch * len(train_dl) + i)

        # 训练一轮，测试准确率一轮
        test_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_acc(test_dl, model, epoch)
    return loss_his

def test_acc(test_dl, model, epoch):
    correct = 0
    total = 0

    # 关闭训练模式
    model.eval()

    with torch.no_grad():
        for data, target in test_dl:
            data,target = data.to(device), target.to(device)
            output = model(data.reshape(-1, 64, 64))
            _,predict = torch.max(output.squeeze(), 1)      # LSTM输出需要修改
            total += target.size(0)
            correct += (predict == target).sum().item()
    print(f'acc: {correct/total*100}%')
    writer.add_scalar("accuracy", correct/total*100, epoch)
    return correct/total*100

if __name__ == '__main__':

    model_rnn = RNN_Classfier()
    model_gru = GRU_Classfier()
    model_lstm = LSTM_Classfier()

    model_list = [model_rnn, model_gru, model_lstm]

    for model in model_list:
        writer = SummaryWriter()
        olt_faces = fetch_olivetti_faces(data_home='face_data', shuffle=True)

        images_tensor = torch.tensor(olt_faces.images)
        target_tensor = torch.tensor(olt_faces.target)

        # dataset拼接
        dataset = [(img, lbl) for img,lbl in zip(images_tensor, target_tensor)]

        train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'

        # 定义损失函数&优化器
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        loss_his = training_data(epochs, model, train_dl)


