#导入必要库
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

#定义RNN模型
class RNN_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size= 28,
            hidden_size=50,
            num_layers=5,
            bias=True,
            batch_first=True
        )
        self.fc = nn.Linear(50,10)
    def forward(self,X):
        output, hidden = self.rnn(X)
        out = self.fc(output[:,-1,:])

        return out

if __name__ == '__main__':
    writer = SummaryWriter()

    #运行到cuda上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #加载数据集
    train_data = MNIST(root='./data',train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root='./data',train=False, transform=ToTensor(), download=True)

    #打包数据集
    train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=64, shuffle=False)

    #实例化模型
    model = RNN_classifier()
    model.to(device)

    #定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(),lr=0.001)

    train_loss = []
    #训练模型
    epochs = 10

    
    for epoch in range(epochs):
        model.train()
        for i, (image,lable) in enumerate(train_dl):
            image, lable = image.to(device), lable.to(device) 
            output = model(image.squeeze())
            loss = loss_fn(output, lable)

            optimzier.zero_grad()
            loss.backward()
             #梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimzier.step()
            train_loss.append(loss.item())
            if i % 100 == 0:
                print(f'epoch:[{epoch+1}/{epochs}] loss:{loss.item():.4f}')
                writer.add_scalar('training loss', loss.item(), epochs * len(train_dl) + i)       

        #模型推理
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for image, lable in test_dl:
                image,lable = image.to(device),lable.to(device)
                out = model(image.squeeze())
                _, predicted = torch.max(out.data, 1)
                total += lable.size(0)
                correct += (predicted == lable).sum().item()
                accuracy = 100 * correct / total
            print(f'epoch:[{epoch+1}/{epochs}]  test accuracy:{accuracy:.4f}%')
            writer.add_scalar('test acc',accuracy, epoch)


    #保存模型
    torch.save(model, 'rnn_model.pth')      #保存全部
    torch.save(model.state_dict(), 'rnn_model_params.pth')     #保存模型参数

    writer.close()

    # #加载模型
    # model = torch.load('rnn_model.pth')

    # #加载模型参数
    # model = RNN_classifier()
    # model.load_state_dict(torch.load('rnn_model_params.pth'))

import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()






