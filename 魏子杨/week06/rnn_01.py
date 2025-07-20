# 1. 实验使用不同的RNN结构，实现一个人脸图像分类器。
# 至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。
# 要求使用tensorboard，提交代码及run目录和可视化截图
import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

class RNN_Classifier(nn.Module):

    def __init__(self,rnn_type):
        super().__init__()
        self.init_rnn(rnn_type)
        self.fc = nn.Linear(256 if 'Bi' in rnn_type else 128, 40)   # 输出层 


    def init_rnn(self, rnn_type):
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=64,   
                hidden_size=128,  
                bias=True,        
                num_layers=2,     
                batch_first=True  
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=64,   
                hidden_size=128,  
                bias=True,        
                num_layers=2,     
                batch_first=True  
            )
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=64,   
                hidden_size=128,  
                bias=True,        
                num_layers=2,     
                batch_first=True  
            )
            #双向
        elif rnn_type == 'BiRNN':
            self.rnn = nn.RNN(
                input_size=64,   
                hidden_size=128,  
                bias=True,        
                num_layers=2,     
                batch_first=True,
                bidirectional=True  
            )
            #双向
        elif rnn_type == 'BiLSTM':
            self.rnn = nn.LSTM(
                input_size=64,   
                hidden_size=128,  
                bias=True,        
                num_layers=2,     
                batch_first=True,
                bidirectional=True  
            )
        else:
            raise ValueError("Unsupported RNN type.")

    def forward(self, x):
        # 输入x的shape为[batch, times, features]
        outputs, l_h = self.rnn(x)  # 连续运算后所有输出值
        # 取最后一个时间点的输出值
        out = self.fc(outputs[:,-1,:])
        return out


if __name__ == '__main__':
    # TensorBoard的走狗
    writer = SummaryWriter()
    #查看是否支持CUDA，支持的话就用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    olivetti_faces = fetch_olivetti_faces(data_home='face_data')
    #图像为64*64
    data = olivetti_faces.images
    labels = olivetti_faces.target


    #打乱顺序并分类，分为训练集与测试集
    # 使用 train_test_split 划分数据集，shuffle=True 划分前会打乱数据，random_state随机种子
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True
    )

    # 转换为 PyTorch 张量
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)


    # 创建数据加载器,先进行封装以便一一对应
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    rnn_types = ['LSTM', 'GRU', 'RNN', 'BiRNN', 'BiLSTM']


    for rnn_type in rnn_types:
        # 实例化模型
        model = RNN_Classifier(rnn_type)
        model.to(device)
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # 训练模型
        num_epochs = 200
        for epoch in range(num_epochs):
            model.train()
            #这里的i就是enumerate搞的鬼，作用是增加一个计数器
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                #去除维度大小为1的数据
                outputs = model(images.squeeze())
                loss = criterion(outputs, labels)
                loss.backward()
                #进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], {rnn_type} Loss: {loss.item():.4f}')
                    writer.add_scalar(f'{rnn_type} training loss', loss.item(), epoch * len(train_loader) + i)
             # 评估模型
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.squeeze())
                    #predicted应该也是个矩阵
                    _, predicted = torch.max(outputs.data, 1)
                    #第0为上有多少数据，也就是有多少行，也就是有多少样本
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], {rnn_type} Test Accuracy: {accuracy:.2f}%')
                writer.add_scalar(f'{rnn_type} test accuracy', accuracy, epoch)


    # 保存全部
    torch.save(model, 'rnn_model.pth')
    # 保存模型参数
    torch.save(model.state_dict(), 'rnn_model_params.pth')

    writer.close()

    # 加载模型
    # model = torch.load('rnn_model.pth')

    # # 加载模型参数
    # model = RNN_Classifier()
    # model.load_state_dict(torch.load('rnn_model_params.pth'))
