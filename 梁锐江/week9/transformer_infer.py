from transformer_train import Seq2SeqTransformer
import torch


def build_vocab(tokens):
    vocab = {'<pad>': 1, '<s>': 2, '</s>': 3}
    idx = 3
    for seq in tokens:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def greedy_decode(model, test_enc, enc_vocab, dec_vocab, inv_dec_vocab, device, max_length=20):
    enc_inputs = torch.tensor([[enc_vocab[token] for token in test_enc]], dtype=torch.long).to(device)
    mem = model.encode(enc_inputs)

    dec_start = torch.tensor([[dec_vocab['<s>']]], dtype=torch.long).to(device)
    for i in range(max_length):
        tgt_mask = torch.triu(torch.zeros(dec_start.size(1), dec_start.size(1)) == 1, diagonal=1).to(device)
        outputs = model.decode(dec_start, mem, tgt_mask)
        out = model.predict(outputs[:, -1, :])
        prob = out.softmax(-1)
        next_token = prob.argmax(-1).item()
        dec_start = torch.cat([dec_start, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        if next_token == dec_vocab['</s>']:
            break
    result = [inv_dec_vocab[idx] for idx in dec_start[0].cpu().numpy()]
    if result[0] == '<s>':
        result = result[1:]
    if '</s>' in result:
        result = result[:result.index('</s>')]
    return ''.join(result)


if __name__ == '__main__':
    corpus = "人生得意须尽欢，莫使金樽空对月"
    chars = list(corpus)

    enc_tokens = []
    dec_tokens = []

    for i in range(1, len(chars)):
        enc = chars[:i]
        dec = ['<s>'] + chars[i:] + ['</s>']

        enc_tokens.append(enc)
        dec_tokens.append(dec)

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
    model = Seq2SeqTransformer(d_model=d_model, enc_vocab_size=enc_voc_size, dec_vocab_size=dec_voc_size,
                               dropout=dropout).to(device)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('transformer_exercise.pth'))
    model.eval()

    # 推理示例
    test_enc = list("人生得意")
    output = greedy_decode(model, test_enc, enc_vocab, dec_vocab, inv_dec_vocab, device)
    print(f"输入: {''.join(test_enc)}")
    print(f"输出: {output}")
