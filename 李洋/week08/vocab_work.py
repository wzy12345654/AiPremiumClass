import  json
import pickle

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray


def read_file(file_path_e, file_path_d):
    enc_data, dec_data = [], []
    with (open(file_path_e, 'r',encoding='utf-8') as f,\
        open(file_path_d,'r',encoding='utf-8') as f1):
        lines_e = f.read().strip().split('\n')
        lines_d = f1.read().strip().split('\n')
        for e_line,d_line in zip(lines_e,lines_d):
            enc_id = e_line.replace(' ','').replace(',','').replace('!`','').replace('?','').replace('。','')
            dec_id = d_line.replace(' ','').replace(',','').replace('!`','').replace('?','').replace('。','')
            dec_id = ['BOS']+list(dec_id)+['EOS']
            enc_data.append(list(enc_id))
            dec_data.append(dec_id)
        assert  len(enc_data) == len(dec_data),'长度不一致'
        return enc_data,dec_data

class Vocab_data(object):
    def __init__(self,vocab):
        self.vocab = vocab

    @classmethod
    def from_json(cls,data):
        desc =  set()
        for d in data:
            desc.update(d)
        token = ['PAD','UNK'] + list(desc)
        vocab = {tk:i for i,tk in enumerate(token)}
        return cls(vocab)



if __name__ =="__main__":
    file_train_e = '../couplet/train/in.txt'
    file_train_d = '../couplet/train/out.txt'
    file_test_e = '../couplet/test/in.txt'
    file_test_d = '../couplet/test/out.txt'

    enc_train,dec_train = read_file(file_train_e,file_train_d)
    enc_test, dec_test = read_file(file_test_e, file_test_d)
    enc_train_vocab = Vocab_data.from_json(enc_train)
    dec_train_vocab = Vocab_data.from_json(dec_train)
    enc_test_vocab = Vocab_data.from_json(enc_test)
    dec_test_vocab = Vocab_data.from_json(dec_test)

    with open('../couplet/enc_data_train.json','w') as f:
        json.dump(enc_train,f)

    with open('../couplet/dec_data_train.json','w') as f:
        json.dump(dec_train,f)

    with open('../couplet/enc_data_test.json','w') as f:
        json.dump(enc_test,f)

    with open('../couplet/dec_data_test.json','w') as f:
        json.dump(dec_test,f)

    with open('../couplet/vocab_train.pkl', 'wb') as f:
        pickle.dump((enc_train_vocab.vocab,dec_train_vocab.vocab),f)

    with open('../couplet/vocab_test.pkl', 'wb') as f:
        pickle.dump((enc_test_vocab.vocab,dec_test_vocab.vocab),f)
