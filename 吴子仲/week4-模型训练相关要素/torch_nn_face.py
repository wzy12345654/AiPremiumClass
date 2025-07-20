import torch
import torch.nn as nn

class TorchNN_face(nn.Module):
    # 初始化
    def __init__(self):     #self 指代新创建模型对象
        super().__init__()

        self.linear1 = nn.Linear(4096, 8192)
        self.bn1 = nn.BatchNorm1d(8192) # 归一化
        self.linear2 = nn.Linear(8192, 16384)
        self.bn2 = nn.BatchNorm1d(16384)
        self.linear3 = nn.Linear(16384, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.linear4 = nn.Linear(1024, 40)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()    # 正则化
    
    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.act(out)
        out = self.drop(out)
        finalOut = self.linear4(out)
        return finalOut

class TorchNN_face_no_bn_drop(nn.Module):
    # 初始化
    def __init__(self):     #self 指代新创建模型对象
        super().__init__()

        self.linear1 = nn.Linear(4096, 8192)
        self.linear2 = nn.Linear(8192, 16384)
        self.linear3 = nn.Linear(16384, 1024)
        self.linear4 = nn.Linear(1024, 40)
        self.act = nn.ReLU()
    
    def forward(self, input_tensor):
        out = self.linear1(input_tensor)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        finalOut = self.linear4(out)
        return finalOut

if __name__ == '__main__':
    model = TorchNN_face()
    model2 = TorchNN_face_no_bn_drop()