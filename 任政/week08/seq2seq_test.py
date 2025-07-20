import torch
import pickle
from EncoderDecoder_concat import Seq2Seq


if __name__ == '__main__':
    # 加载模型
    model = torch.load('seq2seq_model.pth')
    # 加载词汇表
    with open('cvocab.bin' , 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

    # 构建模型
    modelS = Seq2Seq(
        enc_emb_size = len(enc_vocab),
        dec_emb_size = len(dec_vocab),
        emb_dim = 256,
        hidden_size = 256,
        dropout = 0.5
    )
    # 加载模型参数
    modelS.load_state_dict(torch.load('seq2seq_model.pth'))
    # 创建解码器反向字典
    dec_vocab_inv = {v:k for k , v in dec_vocab.items()}

    # 准备输入数据
    input_text = "一 句 相 思 吟 岁 月"
    input_tokens = input_text.split()
    # 把语句弄成序列
    input_ids = [enc_vocab[token] for token in input_tokens]
    # 把序列变成张量
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    # 模型推理
    # 设置最大解码长度
    max_length = 50
    with torch.no_grad():
        # 设置模型为评估模式
        modelS.eval()
        # 解码器
        hidden_state , enc_output = modelS.encoder(input_tensor)

        # 解码器输入shape[1,1]
        dec_input = torch.tensor([[dec_vocab['BOS']]])
        # 输出序列
        dec_tokens = []
        # 循环解码
        while True:
            if len(dec_tokens) >= max_length:
                break
            logits , hidden_state = modelS.decoder(dec_input , hidden_state , enc_output)

            # 预测下一个token index
            next_token = torch.argmax(logits , dim=-1)

            if dec_vocab_inv[next_token.item()] == 'EOS':
                break
            dec_tokens.append(next_token.squeeze().item())
            dec_input = next_token
            hidden_state = hidden_state.view(1 , -1)

        # 输出翻译结果
        print('翻译结果：', ' '.join([dec_vocab_inv[token] for token in dec_tokens]))






