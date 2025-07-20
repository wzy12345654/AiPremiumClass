from attr.validators import max_len as max_len_validator  # 避免变量名冲突
from EncoderDecoderAttention import Sqe2seq
import torch
import pickle as pkl

if __name__ == "__main__":
    state_dict = torch.load('../couplet/model.pth')

    with open('../couplet/vocab_train.pkl', 'rb') as f:
        enc_vocab, dec_vocab = pkl.load(f)

    model = Sqe2seq(len(enc_vocab), len(dec_vocab), 100, 120).cuda()
    model.load_state_dict(state_dict)

    dec_inv = {tk: i for i, tk in dec_vocab.items()}

    enc_inp = '晨露润花花更红'

    enc_out = torch.tensor([[enc_vocab.get(i, enc_vocab['UNK']) for i in list(enc_inp)]]).cuda()

    max_length = len(enc_inp)  # 避免和导入的 max_len 冲突

    model.eval()
    with torch.no_grad():
        hidden_e, out = model.encoder(enc_out)

        dec_s = torch.tensor([[dec_vocab['BOS']]]).cuda()

        dec_token = []
        while True:
            if len(dec_token) >= max_length:
                break
            logits, hidden_d = model.decoder(dec_s, hidden_e, out)

            max_logits = torch.argmax(logits, dim=-1)

            if max_logits.squeeze(0).item() == dec_vocab['EOS']:  # 修正比较逻辑
                break
            dec_token.append(max_logits.squeeze(0).item())

            dec_s = max_logits

            hidden_e = hidden_d.squeeze(0)

    print(''.join([dec_inv[i] for i in dec_token]))