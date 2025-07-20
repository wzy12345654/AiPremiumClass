import pickle
import torch
import torch.nn as nn
from EncodeDecodeAttenModel import Seq2Seq

if __name__ == "__main__":
    # 加载训练好的模型和词典
    state_dict = torch.load("seq2seq_model.pth")    
    with open("vocab.pkl", "rb") as f:
        enc_vocab, dec_vocab = pickle.load(f)
    
    model = Seq2Seq(
        enc_input_dim = len(enc_vocab),
        dec_input_dim = len(dec_vocab),
        emb_dim = 128,
        hidden_dim = 128,
        dropout = 0.5
    )

    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dec_vocab_inv = {v: k for k, v in dec_vocab.items()}

    # 用户输入
    enc_input = "I love you"
    enc_idx = torch.tensor([[enc_vocab.get(word, enc_vocab['UNK']) for word in enc_input.split()]])

    print(enc_idx.shape)

    # 最大解码长度
    max_dec_len = 20

    model.eval()

    with torch.no_grad():
        # encoder last hidden state
        hidden_state, enc_outputs = model.encoder(enc_idx)

        # decoder input
        dec_input = torch.tensor([[dec_vocab['BOS']]])

        # 存储解码结果
        dec_tokens = []
        # 循环decoder
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
            # decoder
            logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)

            # next token index
            next_token = torch.argmax(logits, dim=-1)

            if dec_vocab_inv[next_token.squeeze().item()] == 'EOS':
                break
            
            # 收集每次token_index
            dec_tokens.append(next_token.squeeze().item())
            # 更新decoder输入
            dec_input = next_token

            hidden_state = hidden_state.view(1, -1)

        print("解码结果:", ''.join([dec_vocab_inv[i] for i in dec_tokens]))


