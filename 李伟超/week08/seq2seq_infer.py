import torch
import pickle
from EncoderDecoderAttenModel import Seq2Seq

if __name__ == '__main__':
    state_dict = torch.load('seq2seq_state.bin', map_location='cpu')
    with open('vocab.bin','rb') as f:
        evoc, dvoc = pickle.load(f)

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dvoc_inv = {v:k for k,v in dvoc.items()}

    enc_input = list("晚风摇树树还挺")  # 示例上联
    enc_idx = torch.tensor([[evoc.get(tk, evoc['UNK']) for tk in enc_input]])

    max_dec_len = 50
    with torch.no_grad():
        hidden_state, enc_outputs = model.encoder(enc_idx)
        dec_input = torch.tensor([[dvoc['BOS']]])
        dec_tokens = []
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
            logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)
            next_token = torch.argmax(logits, dim=-1)
            token_str = dvoc_inv[next_token.item()]
            if token_str == 'EOS':
                break
            dec_tokens.append(token_str)
            dec_input = next_token

    print(''.join(dec_tokens))
