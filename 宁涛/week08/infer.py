import torch
import pickle
import random
from process import read_data, get_proc, Vocabulary
from EncoderDecoderAttenModel import Seq2Seq

if __name__ == '__main__':

    enc_data,dec_data = read_data('couplet/test/in.txt', 'couplet/test/out.txt')

    # 加载训练好的模型和词典
    state_dict = torch.load('seq2seqadd_state.bin')
    vocab_file = 'couplet/vocabs'
    vocab = Vocabulary.from_file(vocab_file)

    model = Seq2Seq(
        enc_emb_size=len(vocab.vocab),
        dec_emb_size=len(vocab.vocab),
        emb_dim=200,
        hidden_size=250,
        dropout=0.5,
        state_type='add',
    )
    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dvoc_inv = {v:k for k,v in vocab.vocab.items()}

    # 随机选取测试样本
    rnd_idx = random.randint(0, len(enc_data))
    enc_input = enc_data[rnd_idx]
    dec_output = dec_data[rnd_idx]
    
    enc_idx = torch.tensor([[vocab.vocab[tk] for tk in enc_input]])

    print(enc_idx.shape)

    # 推理
    # 最大解码长度
    max_dec_len = len(enc_input)

    model.eval()
    with torch.no_grad():
        # 编码器输出
        # hidden_state = model.encoder(enc_idx)
        hidden_state, enc_outputs = model.encoder(enc_idx)  # attention

        # 解码器输入 shape [1,1]
        dec_input = torch.tensor([[vocab.vocab['<s>']]])  # <s> token index])

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

            if dvoc_inv[next_token.squeeze().item()] == '</s>':
                break
            # 收集每次token_index 【解码集合】
            dec_tokens.append(next_token.squeeze().item())
            # decoder的下一个输入 = token_index
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)

    # 输出解码结果
    print(f'上联：',''.join(enc_input))
    print("模型预测下联：", ''.join([dvoc_inv[tk] for tk in dec_tokens]))
    print("真实下联：", ''.join(dec_output))
