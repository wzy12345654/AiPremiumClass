import couplets_model
import pickle

if __name__ == '__main__':
    with  open('couplets_vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)

