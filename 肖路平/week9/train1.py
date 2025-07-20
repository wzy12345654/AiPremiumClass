import torch
import torch.nn as nn
if __name__ == '__main__':
        
    # 输入形状: (batch_size=4, num_classes=3)
    logits = torch.tensor([
                        [1.2, 0.5, -0.3], 
                        [0.8, 2.1, -1.0],
                        [-0.5, 1.7, 0.2], 
                        [0.0, 0.0, 0.0]
                        ])

    # 目标标签: (batch_size=4)
    targets = torch.tensor([1, 0, 2, 1])  # 每个样本的真实类别索引

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)
    print(loss)  # 输出标量损失值