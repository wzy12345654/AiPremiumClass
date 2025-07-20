import pickle
import torch
import torch.nn as nn
from EncodeDecodeAttenModel import Seq2Seq

# 读取测试数据
def read_test_data(test_in_file, test_out_file):
    test_enc_data, test_dec_data = [], []
    with open(test_in_file, "r", encoding="utf-8") as f:
        enc_lines = f.read().split("\n")
    with open(test_out_file, "r", encoding="utf-8") as f:
        dec_lines = f.read().split("\n")
    for i in range(len(enc_lines)):
    # 去除标点和空格
        enc_lines[i] = enc_lines[i].replace("。", "").replace("，", "").replace("！", "").replace("？", "").replace(" ", "")
        dec_lines[i] = dec_lines[i].replace("。", "").replace("，", "").replace("！", "").replace("？", "").replace(" ", "")

        test_enc_data.append(enc_lines[i])
        test_dec_data.append(dec_lines[i])
    return test_enc_data, test_dec_data

if __name__ == "__main__":
    # 加载训练好的模型和词典
    state_dict = torch.load("seq2seq_couplet_model.pth")    
    with open("couplet/vocab.pkl", "rb") as f:
        enc_vocab, dec_vocab = pickle.load(f)
    
    model = Seq2Seq(
        enc_input_dim = len(enc_vocab),
        dec_input_dim = len(dec_vocab),
        emb_dim = 64,
        hidden_dim = 128,
        dropout = 0.5
    )

    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dec_vocab_inv = {v: k for k, v in dec_vocab.items()}

    # 读取测试数据
    test_enc_data, test_dec_data = read_test_data("couplet/test/in.txt", "couplet/test/out.txt")
    
    # 取前100条数据
    test_enc_data = test_enc_data[:3]
    for i in range(len(test_enc_data)):
        # 测试输入
        enc_input = test_enc_data[i]
        enc_idx = torch.tensor([[enc_vocab.get(word, enc_vocab['UNK']) for word in list(enc_input)]])

        # print(enc_idx.shape)

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
                print(torch.max(logits, dim=-1))

                if dec_vocab_inv[next_token.squeeze().item()] == 'EOS':
                    break
                
                # 收集每次token_index
                dec_tokens.append(next_token.squeeze().item())
                # 更新decoder输入
                dec_input = next_token

                hidden_state = hidden_state.view(1, -1)

            # 不知为何只输出同一个字
            print("输入:", test_enc_data[i],"\n"
                  "解码结果:", ''.join([dec_vocab_inv[i] for i in dec_tokens]),
                  "真实结果:", test_dec_data[i])