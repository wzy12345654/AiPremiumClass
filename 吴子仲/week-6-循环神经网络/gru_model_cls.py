import torch.nn as nn

class GRU_Classfier(nn.Module):
    def __init__(self,):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=64,      # x的特征维度
            hidden_size=100,     # 隐藏层神经元数量 w_ht[64, 100], w_hh[100, 500]
            bias=True,
            batch_first=True
        )

        self.fc = nn.Linear(100, 40)

    def forward(self, X):
        # X.shape(batch, seq, features)
        outputs, l_h = self.rnn(X)
        # 取最后一个时间点输出值
        out = self.fc(l_h[0])
        return out


