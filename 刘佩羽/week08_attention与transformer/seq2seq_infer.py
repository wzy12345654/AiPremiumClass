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
from EncoderDecoderAttenModel import Seq2Seq

if __name__ == '__main__':
    # 加载训练好的模型和词典
    model = input("请输入模型encoder hidden state 处理的类型（cat or add）：")
    if model == 'cat':
        state_dict = torch.load('seq2seq_state_cat.bin')
    elif model == 'add':
        state_dict = torch.load('seq2seq_state_add.bin')
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
        encoder_hidden_state=model
    )
    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dvoc_inv = {v:k for k,v in dvoc.items()}

    def predict(enc_input):
        # 用户输入
        # enc_input = "晚 风 摇 树 树 还 挺"
        enc_idx = torch.tensor([[evoc.get(tk, evoc['UNK']) for tk in enc_input.split()]], dtype=torch.long)

        # print(enc_idx.shape)

        # 推理
        # 最大解码长度
        max_dec_len = 50

        model.eval()
        with torch.no_grad():
            # 编码器
            # hidden_state = model.encoder(enc_idx)
            hidden_state, enc_outputs = model.encoder(enc_idx)  # attention

            # 解码器输入 shape [1,1]
            dec_input = torch.tensor([[dvoc['BOS']]])

            # 循环decoder
            dec_tokens = []
            while True:
                if len(dec_tokens) >= max_dec_len:
                    break
                # 解码器 
                # logits: [1,1,dec_voc_size]
                # logits,hidden_state = model.decoder(dec_input, hidden_state)
                logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)
                
                # 下个token index
                next_token = torch.argmax(logits, dim=-1)

                if dvoc_inv[next_token.squeeze().item()] == 'EOS':
                    break
                # 收集每次token_index 【解码集合】
                dec_tokens.append(next_token.squeeze().item())
                # decoder的下一个输入 = token_index
                dec_input = next_token
                hidden_state = hidden_state.view(1, -1)
                # 编写计算模型的准确率
        with open('predict_result_cat.txt', 'a', encoding='utf-8') as f:
            f.write(''.join([dvoc_inv[tk] for tk in dec_tokens]))
            f.write('\n')
        # if model == 'cat':
        #     # 输出解码结果,写入文件
        #     with open('predict_result_cat.txt', 'a', encoding='utf-8') as f:
        #         f.write(''.join([dvoc_inv[tk] for tk in dec_tokens]))
        #         f.write('\n')
        # elif model == 'add':
        #     # 输出解码结果,写入文件
        #     with open('predict_result_add.txt', 'a', encoding='utf-8') as f:
        #         f.write(''.join([dvoc_inv[tk] for tk in dec_tokens]))
        #         f.write('\n')
        # print(''.join([dvoc_inv[tk] for tk in dec_tokens]))
    
    # # test 数据整理
    # with open('./couplet/test/in.txt', 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i % 100 == 0:
    #             print(i)
    #         predict(line.strip())
    predict('风 弦 未 拨 心 先 乱 ')