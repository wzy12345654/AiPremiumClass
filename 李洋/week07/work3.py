import jieba
import torch
from spm_work import Emb1
import sentencepiece as spm
from work2 import Emb

embedding_size = 200
hidden_size = 150
num_factory = 2
layers =2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = torch.load('spm_dmc_vocab.pth')

vocab1 = torch.load('commrnt_vocab.pth')

sp = spm.SentencePieceProcessor(model_file='dmc_model.model')

comment1 = '这部剧真的很好看，很精彩'
comment2 = '这部剧好好好好好好难看啊'

#测试结巴分词器使用
comment1_jieba_idx = torch.tensor([vocab1.get(work,vocab1['UNK']) for work in jieba.lcut(comment1)])
comment2_jieba_idx = torch.tensor([vocab1.get(work,vocab1['UNK']) for work in jieba.lcut(comment2)])

#测试spm分词器使用
comment1_spm_idx = torch.tensor([vocab.get(work,vocab['<unk>']) for work in sp.EncodeAsPieces(comment1)])
comment2_spm_idx = torch.tensor([vocab.get(work,vocab['<unk>']) for work in sp.EncodeAsPieces(comment2)])

comment1_jieba_idx = comment1_jieba_idx.unsqueeze(0).to(device)
comment2_jieba_idx = comment2_jieba_idx.unsqueeze(0).to(device)

comment1_spm_idx = comment1_spm_idx.unsqueeze(0).to(device)
comment2_spm_idx = comment2_spm_idx.unsqueeze(0).to(device)
#spm 模型
model_spm = Emb1(len(vocab), embedding_size, hidden_size, num_factory,layers)
model_spm.load_state_dict(torch.load('spm_dmc_model.pth'))
model_spm.to(device)
#jieba 模型
model_jieba = Emb(len(vocab1), embedding_size, hidden_size, num_factory)
model_jieba.load_state_dict(torch.load('coment_emb.pth'))
model_jieba.to(device)

pred1_spm = model_spm(comment1_spm_idx)
pred2_spm = model_spm(comment2_spm_idx)

pred_jieba1 = model_jieba(comment1_spm_idx)
pred_jieba2 = model_jieba(comment2_jieba_idx)

pred_jieba1 = torch.argmax(pred_jieba1, dim=1).item()
pred_jieba2 = torch.argmax(pred_jieba2, dim=1).item()

pred_spm1 = torch.argmax(pred1_spm, dim=1).item()
pred_spm2 = torch.argmax(pred2_spm, dim=1).item()

print('JIEBA分词器结果：')
print(f'影评:{comment1},分类:{ "positive" if pred_jieba1 == 1 else "negative"}')
print(f'影评:{comment2},分类:{ "positive" if pred_jieba2 == 1 else "negative"}')
print()
print('SPM分词器结果')
print(f'影评:{comment1},分类:{ "positive" if pred_spm1 == 1 else "negative"}')
print(f'影评:{comment2},分类:{ "positive" if pred_spm2 == 1 else "negative"}')
