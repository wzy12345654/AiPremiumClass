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
from exercise_transformer_model import Seq2SeqTransformer
import pickle

if __name__ == '__main__':
    d_model = 512
    model_state = torch.load("transfomer_model.pth")
    with open('F:/githubProject/AiPremiumClass/梁锐江/week8/couple_vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    model = Seq2SeqTransformer(d_model=d_model,
                               enc_vocab_size=len(enc_vocab),
                               dec_vocab_size=len(dec_vocab),
                               dropout=0.5)

    rever_vocab = {idx: word for word, idx in dec_vocab.items()}

    user_input = "腾 飞 上 铁 ， 锐 意 改 革 谋 发 展 ， 勇 当 千 里 马 "
    max_length = 50

    model.eval()
    with torch.no_grad():
        enc_mem = model.encode(user_input)
        dec_inputs = torch.tensor([dec_vocab['BOS']])

        while True:

            dec_len = dec_inputs.size(1)
            mask = torch.triu(torch.ones(dec_len, dec_len) == 1, diagonal=1)
            output = model.decode(dec_inputs, enc_mem, mask)

            next_token = torch.argmax(output, dim=-1)

            if next_token.item() == dec_vocab['EOS']:
                break
            dec_inputs.add(next_token)

    print(''.join([rever_vocab[idx] for idx in dec_inputs]))