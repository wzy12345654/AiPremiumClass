import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class CommentsClassifier(nn.Module):

    def __init__(self, vocab_size, emb_size, rnn_hidden_size, num_labels):
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # 字典大小
            embedding_dim=emb_size,  # 词向量大小
            padding_idx=0,  # 填充值索引
        )

        # rnn layer
        self.rnn = nn.LSTM(
            input_size=emb_size,  # 输入特征维度
            hidden_size=rnn_hidden_size,  # 隐藏层状态维度
            batch_first=True,  # 输入和输出的batch维度在第一维
        )

        # 分类预测
        self.classifier = nn.Linear(
            in_features=rnn_hidden_size,  # 输入特征维度
            out_features=num_labels,  # 输出标签数量
        )

    def forward(self, input_data):
        # [batch_size, seq_len] -> [batch_size, seq_len, emb_size]
        out = self.embedding(input_data)
        output, _ = self.rnn(out)
        return self.classifier(output[:, -1, :])  # 取最后一个时间步的输出进行分类


if __name__ == "__main__":
    input_data = torch.randint(1, 10, size=(10, 20))  # 模拟输入数据
    model = CommentsClassifier(
        vocab_size=10, emb_size=20, rnn_hidden_size=30, num_labels=2
    )
    logits = model(input_data)
    print(logits.shape)  # 输出形状应为 [batch_size, num_labels]
