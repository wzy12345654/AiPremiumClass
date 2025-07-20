import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader




def read_data(in_data,out_data):
    """
      读取训练数据返回数据集合
    """
    in_datas,out_datas = [],[]
    in_ = open(in_data, encoding='utf-8')
    out_ = open(out_data, encoding='utf-8')
    
    for in_line,out_line in zip(in_,out_):
      # 去除换行符
      in_datas.append(in_line.split())
      out_datas.append(out_line.split())
      
    assert len(in_datas) == len(out_datas),'编码数据与解码数据长度不一致！'
    return in_datas,out_datas
class Vocabulary: 
    def __init__(self, vocab):
      self.vocab = vocab
    
    @classmethod
    def from_documents(cls,vocab_file):
      # 字典构建 字符为token，词汇为token
      with open(vocab_file,'r',encoding='utf-8') as f:
        vocab = f.read().split('\n')
        vocab = ['<pad>'] + [tk for tk in vocab if tk!=''] 
      return cls({tk:i for i,tk in enumerate(vocab)})
        
        
def get_proc(vocab):
   
    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用介乎
    def batch_proc(data):
      """
        批次数据处理并返回
      """
      in_ids,out_ids,labels = [],[],[]
      for eoc,dec in data:
        # token -> token index
        in_idx = [ vocab['<s>']]+ [vocab[tk] for tk in eoc] + [ vocab['</s>']]
        out_idx =[ vocab['<s>']]+  [vocab[tk] for tk in dec] + [ vocab['</s>']]
        
        in_ids.append(torch.tensor(in_idx))
        out_ids.append(torch.tensor(out_idx[:-1]))
        labels.append(torch.tensor(out_idx[1:]))
      # 数据转换成张量 [batch, max_token_len]
      enc_input = pad_sequence(in_ids, batch_first=True)
      dec_input = pad_sequence(out_ids, batch_first=True)
      targets = pad_sequence(labels, batch_first=True)
      # 返回数据都是模型训练和推理的需要
      return enc_input,dec_input,targets

    return batch_proc
     
     
if __name__ =='__main__':
    vocab_file = '../data/couplet/vocabs'
    vocab = Vocabulary.from_documents(vocab_file)
    
    # 训练数据
    in_train_file='../data/couplet/train/in.txt'
    out_train_file='../data/couplet/train/out.txt'
    in_data,out_data = read_data(in_train_file,out_train_file)
    
    print('enc length', len(in_data))
    print('dec length', len(out_data))
    
    print('词汇表',len(vocab.vocab))
    # 训练样本
    dataset = list(zip(in_data,out_data))
    
    dataloader = DataLoader(dataset,batch_size=2,shuffle=True, collate_fn=get_proc(vocab.vocab))
    # 数据整体json数据集（json）
    import json
    with open('../data/couplet/encoder.json','w',encoding='utf-8') as f:
        json.dump(in_data,f)
        
    with open('../data/couplet/decoder.json','w',encoding='utf-8') as f:
        json.dump(out_data,f)

    # with open('../data/couplet/encoders.json','w',encoding='utf-8') as f:
    #    for enc in in_data:
    #       str_json = json.dumps(enc)
    #       f.write(str_json+'\n')
     # # 数据每行都是json数据（jsonl）
    # with open('encoders.json', 'w', encoding='utf-8') as f:
    #     for enc in enc_data:
    #         str_json = json.dumps(enc)
    #         f.write(str_json + '\n')