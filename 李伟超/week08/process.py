import torch
from torch.nn.utils.rnn import pad_sequence

def read_duilian_data(in_file, out_file):
    enc_data, dec_data = [], []
    with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'r', encoding='utf-8') as f_out:
        for x, y in zip(f_in, f_out):
            enc_data.append(list(x.strip()))
            dec_data.append(['BOS'] + list(y.strip()) + ['EOS'])
    return enc_data, dec_data

def get_proc(enc_voc, dec_voc):
    def batch_proc(data):
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]
            enc_ids.append(torch.tensor(enc_idx))
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            labels.append(torch.tensor(dec_idx[1:]))
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)
        return enc_input, dec_input, targets
    return batch_proc

class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        tokens = ['PAD','UNK','BOS','EOS']
        for doc in documents:
            for ch in doc:
                if ch not in tokens:
                    tokens.append(ch)
        vocab = {tk: i for i, tk in enumerate(tokens)}
        return cls(vocab)
