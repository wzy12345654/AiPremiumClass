import torch
import torch.nn as nn

class faceNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 1024)
        self.bn1 = nn.BatchNorm1d(2048)
        self.linear2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, output_tensor):
        out = self.linear1(output_tensor)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear3(out)
        
        return out
    
if __name__ == "__main__":
    model = faceNN()
    print(model)
    