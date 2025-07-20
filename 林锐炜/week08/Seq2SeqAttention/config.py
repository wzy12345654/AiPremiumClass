class Config:
    def __init__(self):
        self.enc_hidden_mode = {
            'num_layers': 2,  # 须>1才能使用dropout
            'bidirectional': True,
            'merge_mode': 'concat'
        }
    # 数据路径
    enc_data_path = "in.txt"
    dec_data_path = "out.txt"
    enc_vocab_path = "enc_vocab.pkl"
    dec_vocab_path = "dec_vocab.pkl"
    model_save_path = "seq2seq_couplet.pt"

    # 模型参数
    emb_dim = 256
    hidden_size = 512
    dropout = 0.3
    enc_hidden_mode = "concat"  # 可选concat/add

    # 训练参数
    batch_size = 128
    lr = 1e-3
    epochs = 20
    max_length = 50  # 最大生成长度
