"""
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""
import torch
import pickle
from hw_EncoderDecoderAttention import Seq2Seq
from torch.utils.data import DataLoader
from hw_process import get_proc
import random

if __name__ == '__main__':
    # 1.加载数据集 & 定义模型、损失函数和优化器
    with open('data/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    # with open('data/vocab.bin','rb') as f:
    #     vocab = pickle.load(f)
    with open('data/chinese-couplets-vocab.bin','rb') as f:
        vocab = pickle.load(f)

    # 加载模型的参数
    state_dict = torch.load('data/model/seq2seq_model.bin', map_location=torch.device('cpu'))
    model = Seq2Seq(
        enc_emb_size=len(vocab),
        dec_emb_size=len(vocab),
        emb_dim=128,
        hidden_size=256,
        dropout=0.5,
    )
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
        enc_output, enc_hidden = model.encoder(enc_idx) # enc_hidden: [1, batch_size, hidden_dim]

        # 2. decoder
        dec_input = torch.tensor([[vocab['<s>']]]) # [1,1]
        hidden_state = enc_hidden.view(1, -1) # [1, batch_size, hidden_dim]
        dec_tokens = [] # 解码集合
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
            # 解码器 
            # logits: [1,1,dec_voc_size]
            logits,hidden_state = model.decoder(dec_input, hidden_state,enc_output) # hidden_state: [1, batch_size, hidden_dim]
            # 计算argmax, 的到下一个token_index
            # token_idx = torch.argmax(logits,dim=-1).item() # [1,1] -> [1] -> int
            next_token = torch.argmax(logits, dim=-1)
            if dvoc_inv[next_token.squeeze().item()] == '</s>':
                break
            # 收集每次token_index 【解码集合】
            dec_tokens.append(next_token.squeeze().item())
            # decoder的下一个输入 = token_index
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)
        
        # 输出解码结果
        print('上    联: ',''.join(enc_input))
        print('真实下联: ',''.join(dec_output[1:-1]))
        print("预测下联: ",''.join([dvoc_inv[idx] for idx in dec_tokens]))


            
