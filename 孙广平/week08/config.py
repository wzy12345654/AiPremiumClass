import torch

# 超参数配置
config = {
    "embed_dim": 256,          # 词嵌入维度
    "enc_hid_dim": 512,        # 编码器隐藏层维度
    "dec_hid_dim": 512,        # 解码器隐藏层维度
    "n_layers": 1,             # RNN层数
    "dropout": 0.5,            # Dropout概率
    "bidirectional": True,     # 是否使用双向LSTM
    "batch_size": 64,          # 批处理大小
    "lr": 0.001,               # 学习率
    "clip": 1.0,               # 梯度裁剪阈值
    "n_epochs": 20,            # 训练轮数
    "teacher_forcing_ratio": 0.5,  # 教师强制比例
    "max_vocab_size": 10000,   # 最大词汇表大小
    "max_len": 100,            # 最大序列长度
    "merge_mode": "concat" ,    # 双向LSTM合并模式："concat"/"add"
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
}

