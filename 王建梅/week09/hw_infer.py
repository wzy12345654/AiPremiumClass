# Seq2SeqTransformer model 推理

import torch
import pickle
import hw_seq2seqTransformer as est
import random

if __name__ == '__main__':
    # 1.加载数据集 & 定义模型、损失函数和优化器
    with open('data/model/hw_test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    with open('data/model/hw_vocab.bin','rb') as f:
        vocab = pickle.load(f)

    # 加载模型的参数
    # 超参数设置
    num_encoder_layers = 2
    num_decoder_layers = 2
    emb_size = 128
    nhead = 2
    dim_feedforward = 256  # 前馈神经网络的隐藏层维度
    src_vocab_size = len(vocab)  # 输入词典大小
    tgt_vocab_size = len(vocab)  # 输出词典大小

    state_dict = torch.load('data/model/hw_seq2seqTransformer_model.bin', map_location=torch.device('cpu'))
    model = est.Seq2SeqTransformer(num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                         emb_size=emb_size, nhead=nhead, 
                                         src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
                                         dim_feedforward=dim_feedforward)
    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dvoc_inv = {v:k for k,v in vocab.items()}

    # 随机抽取测试样本
    rnd_idx = random.randint(0, len(test_dataset))
    enc_input = test_dataset[rnd_idx][0]  
    dec_output = test_dataset[rnd_idx][1]
    enc_idx = torch.tensor([[vocab[tk] for tk in enc_input]])
    print(enc_idx.shape)

    # 推理流程
    # 最大解码长度
    max_dec_len = len(enc_input)

    model.eval()
    with torch.no_grad():
        # 1. encoder
        enc_output = model.encode(enc_idx) # enc_hidden: [1, batch_size, hidden_dim]

        # 2. decoder
        dec_input = torch.tensor([[vocab['<s>']]]) # [1,1]
        
        dec_tokens = [] # 解码集合
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
            # 解码器 
            # logits: [1,1,dec_voc_size]
            logits = model.decode(dec_input, enc_output,None) # hidden_state: [1, batch_size, hidden_dim]
            out = model.generator(logits)[:, -1, :]  # 取最后一个时间步           
            prob = out.softmax(-1)
            next_token = prob.argmax(-1).item()
            # 计算argmax, 的到下一个token_index        
            if dvoc_inv[next_token] == '</s>':
                break
            # 收集每次token_index 【解码集合】
            dec_tokens.append(next_token)
            # decoder 两个token连接在一起
            dec_input = torch.cat([dec_input, torch.tensor([[next_token]])], dim=1)
        
        # 输出解码结果
        print('上    联: ',''.join(enc_input))
        print('真实下联: ',''.join(dec_output[1:-1]))
        print("预测下联: ",''.join([dvoc_inv[idx] for idx in dec_tokens]))
