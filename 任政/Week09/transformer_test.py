import torch
from transformer_model import Seq2SeqTransformer
from transformer_train import generate_square_subsequent_mask , build_vocab

# 生成解码器
def generate(model, enc_input, enc_vocab , dec_vocab , inv_dec_vocab , device , max_length=100):
    model.eval()
    # 1. 输入序列
    enc_input = torch.tensor([enc_vocab[t] for t in enc_input], dtype=torch.long).unsqueeze(0).to(device)
    # 构建mask
    enc_pad_mask = (enc_input == 0)
    memory = model.encode(enc_input,)
    ys = torch.tensor([[dec_vocab['<s>']]], dtype=torch.long).to(device)
    for i in range(max_length):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        dec_pad_mask = (ys == 0)
        out = model.decode(ys, memory, tgt_mask)
        # 取最后一个token
        out = model.predict(out[:, -1])
        prob = out.softmax(dim=-1)
        next_token = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        if next_token == dec_vocab['</s>']:
            break
    # 去掉<s>和</s>
    return [inv_dec_vocab[t.item()] for t in ys[0] if t not in [dec_vocab['<s>'], dec_vocab['</s>']]]



if __name__ == '__main__':
    # 模型数据
    corpus = "君不见，黄河之水天上来，奔流到海不复回。君不见，高堂明镜悲白发，朝如青丝暮成雪。"
    chs = list(corpus)
    enc_tokens, dec_tokens = [], []
    for i in range(1, len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    # 构建词典
    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)
    inv_dec_vocab = {v: k for k, v in dec_vocab.items()}
    # 模型参数（需要与训练时保持一致）
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)
    # 创建模型实例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size, dec_voc_size).to(device)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('transformer.pth'))
    # 生成
    enc_input = "君不见，高堂明镜悲白发，"
    generated_tokens = generate(model, enc_input, enc_vocab, dec_vocab, inv_dec_vocab, device)
    print("Generated:", ''.join(generated_tokens))

