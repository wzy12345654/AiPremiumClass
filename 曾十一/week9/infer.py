import torch
from transformer_seq2seq_model import Seq2SeqTransformer
from data_fix import Vocabulary
import pickle

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 加载词表
vocab_file = '/mnt/data_1/zfy/self/homework/kagglehub/datasets/jiaminggogogo/chinese-couplets/versions/2/couplet/vocabs'
vocab = Vocabulary.from_file(vocab_file)

# 用来从 id 转回汉字
idx2word = {i: w for w, i in vocab.vocab.items()}

# 加载训练好的模型
model = Seq2SeqTransformer(
    d_model=64,
    nhead=4,
    num_enc_layers=2,
    num_dec_layers=2,
    dim_forward=256,
    dropout=0.1,
    enc_voc_size=len(vocab),
    dec_voc_size=len(vocab)
).to(device)

model.load_state_dict(torch.load('/mnt/data_1/zfy/self/homework/seq2seq_state.bin', map_location=device))
model.eval()

input_text = "人生自古谁无死"

# 加上起始和结束标志
enc_tokens = ['<s>'] + list(input_text) + ['</s>']
# 把每个字转成词表里的 ID
enc_ids = [vocab[tk] for tk in enc_tokens]
# 转成 PyTorch 张量
enc_tensor = torch.tensor(enc_ids).unsqueeze(0).to(device)  # [1, seq_len]


# 初始化 decoder 的输入
dec_ids = [vocab['<s>']]  # 初始是 <s>

max_len = 30  # 最多生成 30 个 token

for _ in range(max_len):
    # 当前 decoder 输入
    dec_tensor = torch.tensor(dec_ids).unsqueeze(0).to(device)  # [1, len]
    
    # 构建 decoder 的 mask（防止看到未来的信息）
    tgt_mask = torch.triu(torch.full((dec_tensor.size(1), dec_tensor.size(1)), float('-inf')), diagonal=1).to(device)

    # 构建 padding mask
    enc_pad_mask = (enc_tensor == vocab['<pad>'])
    dec_pad_mask = (dec_tensor == vocab['<pad>'])

    # 调用模型，生成 logits
    with torch.no_grad():
        output = model(enc_tensor, dec_tensor, tgt_mask, enc_pad_mask, dec_pad_mask)

    # 取最后一个位置的输出，找出概率最大的 token
    next_token = output[0, -1].argmax().item()

    if next_token == vocab['</s>']:
        break

    dec_ids.append(next_token)


# 转成汉字，去掉开头的 <s>
result = ''.join([idx2word[i] for i in dec_ids[1:]])
print("生成下联：", result)


