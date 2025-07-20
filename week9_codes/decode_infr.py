import torch
from transformer_model import Seq2SeqTransformer
from train import build_vocab, generate_square_subsequent_mask  # 添加这行
# 如需加载词典等数据结构，也可导入pickle或json
# import pickle
# import json

# 贪婪解码（所有获取结果中，取概率最大的值）
def greedy_decode(model, enc_input, enc_vocab, dec_vocab, inv_dec_vocab, device, max_len=20):
    model.eval()
    enc_input = torch.tensor([[enc_vocab.get(t, 0) for t in enc_input]], dtype=torch.long).to(device)
    enc_pad_mask = (enc_input == 0)
    memory = model.encode(enc_input)
    ys = torch.tensor([[dec_vocab['<s>']]], dtype=torch.long).to(device)
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        dec_pad_mask = (ys == 0)
        out = model.decode(ys, memory, tgt_mask)
        out = model.predict(out)[:, -1, :]  # 取最后一个时间步
        prob = out.softmax(-1)
        next_token = prob.argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        if next_token == dec_vocab['</s>']:
            break
    # 去掉<s>和</s>
    result = [inv_dec_vocab[idx] for idx in ys[0].cpu().numpy()]
    if result[0] == '<s>':
        result = result[1:]
    if '</s>' in result:
        result = result[:result.index('</s>')]
    return ''.join(result)

if __name__ == '__main__':
    # 加载词典和模型参数
    corpus = "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [], []
    for i in range(1,len(chs)):
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
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, 
                              dim_forward, dropout, enc_voc_size, dec_voc_size).to(device)
    
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('transformer.pth'))
    model.eval()

    # 推理示例
    test_enc = list("人生得意须尽欢，莫使")
    output = greedy_decode(model, test_enc, enc_vocab, dec_vocab, inv_dec_vocab, device)
    print(f"输入: {''.join(test_enc)}")
    print(f"输出: {output}")