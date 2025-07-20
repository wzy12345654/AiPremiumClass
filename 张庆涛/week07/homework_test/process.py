import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader



enc_data,dec_data = [],[]
def read_data(data_file):
    """
      读取训练数据返回数据集合
    """
    with open(data_file, 'r', encoding='utf-8') as f:
      # 读取记录行
     lines = f.read().split("\n")
     for line in lines:
        # 按照空格分割
        if line == '':
          continue
        enc,dec = line.split("\t")
        # 数据清洗
        enc = enc.replace(",","").replace('.','').replace('?','').replace('!','')
        dec = dec.replace("，","").replace('。','').replace('？','').replace('！','')
        
        # 分词
        enc_tks = enc.split()
        dec_tks = ['BOS'] + list(dec) + ['EOS']
        
        # append 数据
        enc_data.append(enc_tks)
        dec_data.append(dec_tks)
    assert len(enc_data) == len(dec_data),'编码数据与解码数据长度不一致！'
    return enc_data,dec_data
class Vocabulary: 
    def __init__(self, vocab):
      self.vocab = vocab
    
    @classmethod
    def from_documents(cls,documents):
      # 字典构建 字符为token，词汇为token
      no_repeat_tokens = set()
      for cmt in documents:
        no_repeat_tokens.update(list(cmt))
      tokens = ['PAD','UNK'] + list(no_repeat_tokens)
      
      vocab = {tk:i for i,tk in enumerate(tokens)}
      return cls(vocab)
        
        
def get_proc(enc_voc, dec_voc):
   
    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用介乎
    def batch_proc(data):
      """
        批次数据处理并返回
      """
      enc_ids,dec_ids,labels = [],[],[]
      for eoc,dec in data:
        # token -> token index
        enc_idx = [enc_voc[token] for token in eoc]
        dec_idx = [dec_voc[token] for token in dec]
        
        enc_ids.append(torch.tensor(enc_idx))
        dec_ids.append(torch.tensor(dec_idx[:-1]))
        labels.append(torch.tensor(dec_idx[1:]))
      # 数据转换成张量 [batch, max_token_len]
      enc_input = pad_sequence(enc_ids, batch_first=True)
      dec_input = pad_sequence(dec_ids, batch_first=True)
      targets = pad_sequence(labels, batch_first=True)
      # 返回数据都是模型训练和推理的需要
      return enc_input,dec_input,targets

    return batch_proc
     
     
if __name__ =='__main__':
    enc_data,dec_data = read_data('../data/cmn.txt')
    print("enc length",len(enc_data))
    print("dec length",len(dec_data))
    
    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)
    
    print("enc 词汇数量",len(enc_vocab.vocab))
    print("dec 词汇数量",len(dec_vocab.vocab))
    
    # 解码+编码 
    dataset = list(zip(enc_data,dec_data))
    
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True, collate_fn=get_proc(enc_vocab,dec_vocab))
    
    import json
    
    # 数据整体json数据集（json）
    with open('../data/encoder.json','w',encoding='utf-8') as f:
        json.dump(enc_data,f)
        
    with open('../data/decoder.json','w',encoding='utf-8') as f:
        json.dump(dec_data,f)

    import pickle
    with open('../data/vocab.bin','wb') as f:
        pickle.dump((enc_vocab.vocab,dec_vocab.vocab),f)
    
    with open('../data/encoders.json','w',encoding='utf-8') as f:
       for enc in enc_data:
          str_json = json.dumps(enc)
          f.write(str_json+'\n')