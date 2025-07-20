# process.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def read_data(data_file1,data_file2):
    """
    读取训练数据返回数据集合
    """

    enc_data,dec_data=[],[]
    with open(data_file1,encoding='utf-8') as f:
        # 读取记录行
        lines = f.read().split('\n')
        for line in lines:
            if line == '':
                continue
            enc=line.strip()
            # 数据清洗
            enc = enc.replace(',','').replace('.','').replace('!','').replace('?','').replace('、','').replace('。','')     
            # 分词
            enc_tks = enc.split()
            # 保存
            enc_data.append(enc_tks)
    with open(data_file2,encoding='utf-8') as f:
        # 读取记录行
        lines = f.read().split('\n')
        for line in lines:
            if line == '':
                continue
            dec=line.strip() 
            dec = dec.replace(',','').replace('。','').replace('？','').replace('！','').replace('、','')   
            # 分词
            dec_tks = ['BOS'] + list(dec) + ['EOS']
            # 保存
            dec_data.append(dec_tks)

    # 断言
    assert len(enc_data) == len(dec_data), '编码数据与解码数据长度不一致！'

    return enc_data, dec_data

def get_proc(enc_voc, dec_voc):

    # 嵌套函数定义
    # 外部函数变量生命周期会延续到内部函数调用结束 （闭包）

    def batch_proc(data):
        """
        批次数据处理并返回
        """
        enc_ids, dec_ids, labels = [],[],[]
        for enc,dec in data:
            # token -> token index
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]

            # encoder_input
            enc_ids.append(torch.tensor(enc_idx))
            # decoder_input
            dec_ids.append(torch.tensor(dec_idx[:-1]))
            # label
            labels.append(torch.tensor(dec_idx[1:]))

        
        # 数据转换张量 [batch, max_token_len]
        # 用批次中最长token序列构建张量
        enc_input = pad_sequence(enc_ids, batch_first=True)
        dec_input = pad_sequence(dec_ids, batch_first=True)
        targets = pad_sequence(labels, batch_first=True)

        # 返回数据都是模型训练和推理的需要
        return enc_input, dec_input, targets

    # 返回回调函数
    return batch_proc    

class Vocabulary:

    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        # 字典构建（字符为token、词汇为token）
        no_repeat_tokens = set()
        for cmt in documents:
            no_repeat_tokens.update(list(cmt))  # token list
        # set转换为list，第0个位置添加统一特殊token
        tokens = ['PAD','UNK'] + list(no_repeat_tokens)

        vocab = { tk:i for i, tk in enumerate(tokens)}

        return cls(vocab)

if __name__ == '__main__':
    
    enc_data,dec_data = read_data('E:/Mi/week08/couplet/test/in.txt','E:/Mi/week08/couplet/test/out.txt')
    
    print('enc length', len(enc_data))
    print('dec length', len(dec_data))

    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)

    print('编码器词汇数量', len(enc_vocab.vocab))
    print('解码器词汇数量', len(dec_vocab.vocab))
    
    # 编码+解码（训练样本）
    dataset = list(zip(enc_data, dec_data))
    # Dataloader

    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=get_proc(enc_vocab.vocab, dec_vocab.vocab)   # callback
    )

    # 数据缓存
    import json

    # 数据整体json数据集（json）
    with open('encoder.json', 'w', encoding='utf-8') as f:
        json.dump(enc_data, f)  
    
    with open('decoder.json', 'w', encoding='utf-8') as f:
        json.dump(dec_data, f)  

    import pickle
    with open('vocab.bin','wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab),f)


    # # 数据每行都是json数据（jsonl）
    # with open('encoders.json', 'w', encoding='utf-8') as f:
    #     for enc in enc_data:
    #         str_json = json.dumps(enc)
    #         f.write(str_json + '\n')
# train.py
import pickle
import torch
import json
from torch.utils.data import DataLoader
from process import get_proc
from EncoderDecoderAttenModel import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练数据
    with open('E:/Mi/week08/vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    with open('E:/Mi/week08/encoder.json') as f:
        enc_data = json.load(f)
    with open('E:/Mi/week08/decoder.json') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data,dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=get_proc(evoc, dvoc))

    # 构建训练模型
    # 模型构建
    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )
    model.to(device)

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(20):
        model.train()
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播 
            logits, _ = model(enc_input, dec_input)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            writer.add_scalar('Loss/train', loss.item(), epoch)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'seq2seq_state.bin')
  # infer.py
  """
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""
import torch
import pickle
from EncoderDecoderAttenModel import Seq2Seq

if __name__ == '__main__':
    # 加载训练好的模型和词典
    state_dict = torch.load('seq2seq_state.bin')
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )
    model.load_state_dict(state_dict)

    # 创建解码器反向字典
    dvoc_inv = {v:k for k,v in dvoc.items()}

    # 用户输入
    enc_input = "水 流 知 入 海"
    # enc_input = "What I'm about to say is strictly between you and me"
    # enc_input = "I used to go swimming in the sea when I was a child"
    enc_idx = torch.tensor([[evoc[tk] for tk in enc_input.split()]])

    print(enc_idx.shape)

    # 推理
    # 最大解码长度
    max_dec_len = 50

    model.eval()
    with torch.no_grad():
        # 编码器
        hidden_state,enc_outputs = model.encoder(enc_idx)
        # hidden_state, enc_outputs = model.encoder(enc_idx)  # attention

        # 解码器输入 shape [1,1]
        dec_input = torch.tensor([[dvoc['BOS']]])

        # 循环decoder
        dec_tokens = []
        while True:
            if len(dec_tokens) >= max_dec_len:
                break
            # 解码器 
            # logits: [1,1,dec_voc_size]
            logits,hidden_state = model.decoder(dec_input, hidden_state,enc_outputs)
            # logits,hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)
            
            # 下个token index
            next_token = torch.argmax(logits, dim=-1)

            if dvoc_inv[next_token.squeeze().item()] == 'EOS':
                break
            # 收集每次token_index 【解码集合】
            dec_tokens.append(next_token.squeeze().item())
            # decoder的下一个输入 = token_index
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)

    # 输出解码结果
    print(''.join([dvoc_inv[tk] for tk in dec_tokens]))
