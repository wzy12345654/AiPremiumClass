#构建encoder和decoder的词典
    # 模型训练数据格式： X：([enc_token_matrix], [dec_token_matrix] shifted right)，y [dec_token_matrix] shifted
    # 1. 通过词典把token转换为token_index
    # 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
    # 3. Dataloader的collate_fn调用自定义转换方法，填充模型训练数据
    #    3.1 encoder矩阵使用pad_sequence填充
    #    3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
    #    3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import pickle
class Vocabulary:
    def __init__(self, vocab):
        self.vocab = vocab
        self.pad_idx = vocab['PAD']
        self.unk_idx = vocab['UNK']
        self.sos_idx = vocab.get('<s>', None)
        self.eos_idx = vocab.get('</s>', None)
    @classmethod
    def from_documents(cls, documents):
        no_repeat_tokens = sorted({tk for doc in documents for tk in doc})
        tokens = ['PAD', 'UNK'] + no_repeat_tokens
        vocab = {tk: i for i, tk in enumerate(tokens)}
        return cls(vocab)
def get_proc(enc_voc, dec_voc):
    def batch_proc(data):
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in data:
            enc_idx = [enc_voc[tk] for tk in enc]
            dec_idx = [dec_voc[tk] for tk in dec]
            enc_ids.append(torch.tensor(enc_idx, dtype=torch.long))
            dec_ids.append(torch.tensor(dec_idx[:-1], dtype=torch.long))
            labels.append(torch.tensor(dec_idx[1:], dtype=torch.long))
        enc_input = pad_sequence(enc_ids, batch_first=True, padding_value=enc_voc['PAD'])
        dec_input = pad_sequence(dec_ids, batch_first=True, padding_value=dec_voc['PAD'])
        targets = pad_sequence(labels, batch_first=True, padding_value=dec_voc['PAD'])
        return enc_input, dec_input, targets
    return batch_proc

if __name__ == '__main__':
    corpus = [
        "人生得意须尽欢，莫使金樽空对月",
        "天生我材必有用，千金散尽还复来",
        "长风破浪会有时，直挂云帆济沧海",
        "举杯邀明月，对影成三人",
        "会当凌绝顶，一览众山小"
    ]
    chs_list = [list(line) for line in corpus]
    enc_tokens, dec_tokens = [], []
    for chs in chs_list:
        for i in range(1, len(chs)):
            enc = chs[:i]
            dec = ['<s>'] + chs[i:] + ['</s>']
            enc_tokens.append(enc)
            dec_tokens.append(dec)
    enc_vocab = Vocabulary.from_documents(enc_tokens)
    dec_vocab = Vocabulary.from_documents(dec_tokens)
    dataset = list(zip(enc_tokens, dec_tokens))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=get_proc(enc_vocab.vocab, dec_vocab.vocab)
    )

    for batch in dataloader:
        enc_input, dec_input, targets = batch
    #  for batch_idx, (enc_input, dec_input, targets) in enumerate(dataloader):
    #      enc_input, dec_input, targets = batch
    #      打印原始数据示例
    #     print("\nSample encoder tokens:", enc_tokens[0])
    #     print("Sample decoder tokens:", dec_tokens[0])
    #     # 打印转换后的索引
    #     print("\nEnc input[0]:", enc_input[0])
    #     print("Dec input[0]:", dec_input[0])
    #     print("Targets[0]:", targets[0])
    #     break

    with open('encoders.json', 'w', encoding='utf-8') as f:
        json.dump(enc_tokens, f,ensure_ascii=False)
    with open('decoders.json', 'w', encoding='utf-8') as f:
        json.dump(dec_tokens, f,ensure_ascii=False)
    # 保存字典（pickle）
    with open('vocabs.bin', 'wb') as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab), f)

    # with open('vocab.bin', 'rb') as f:
    #     enc_vocab, dec_vocab = pickle.load(f)
    # print(enc_vocab)  # 查看 encoder 词表
    # print(dec_vocab)  # 查看 decoder 词表