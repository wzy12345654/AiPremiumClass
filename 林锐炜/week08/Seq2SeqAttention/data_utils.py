import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

class CoupletDataset(Dataset):
    def __init__(self, enc_texts, dec_texts):
        self.enc_texts = enc_texts
        self.dec_texts = dec_texts

    def __len__(self):
        return len(self.enc_texts)

    def __getitem__(self, idx):
        return self.enc_texts[idx], self.dec_texts[idx]


class Vocab:
    def __init__(self, tokens=None):
        self.pad = "<PAD>"  # 保证位置0是PAD（重要）
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        specials = [self.pad, self.unk, self.bos, self.eos]

        self.stoi = {}
        self.itos = {}

        # 强制添加特殊符号到词汇表（不论输入中是否存在）
        idx = 0
        for tok in specials:
            self.stoi[tok] = idx
            self.itos[idx] = tok
            idx += 1

        # 添加其他字符
        if tokens is not None:
            tokens = set(tokens) - set(specials)  # 去除已存在的特殊符号
            for tok in sorted(tokens):
                self.stoi[tok] = idx
                self.itos[idx] = tok
                idx += 1

        self.bos_idx = self.stoi[self.bos]
        self.eos_idx = self.stoi[self.eos]
        self.pad_idx = self.stoi[self.pad]

    def __len__(self):
        return len(self.stoi)

    @classmethod
    def build_from_texts(cls, texts, min_freq=1, add_specials=True):
        """ texts参数示例: [[字1, 字2,...], ...] """
        counter = {}
        # 展开所有字符时进行有效性检查
        valid_chars = set()
        for chars in texts:
            for char in chars:
                if char.strip() == '':  # 过滤掉空格
                    continue
                if len(char) > 1:  # 防止误处理多个字符
                    print(f"存在多余字符: {char} -> 跳过")
                    continue
                valid_chars.add(char)

        return cls(sorted(valid_chars))  # 强制包含特殊符号


def load_data(enc_path, dec_path):  # 新增接受两个输入路径参数
    enc_texts, dec_texts = [], []

    # 验证文件是否存在
    for path in [enc_path, dec_path]:
        if not os.path.exists(path):
            print(f"关键错误: 文件 {os.path.abspath(path)} 不存在！")
            return [], []

    # 读取并同步检查两个文件
    with open(enc_path, 'r', encoding='utf-8') as f_enc, \
            open(dec_path, 'r', encoding='utf-8') as f_dec:

        enc_lines = f_enc.readlines()
        dec_lines = f_dec.readlines()

        # 行数一致性检查
        if len(enc_lines) != len(dec_lines):
            print(f"文件行数不一致! in.txt有{len(enc_lines)}行, out.txt有{len(dec_lines)}行")
            return [], []

        print(f"检查通过: 输入输出文件均有{len(enc_lines)}行")
    # 统一处理数据
    print("\n样本处理过程:")
    for line_num, (enc_line, dec_line) in enumerate(zip(enc_lines, dec_lines), 1):
        try:
            # 处理上联
            enc = enc_line.strip().replace(' ', '')  # 去除空格后处理单个汉字
            if not enc:
                print(f"第{line_num}行上联内容为空，已跳过")
                continue
            enc_texts.append(list(enc))  # "晚风摇树" → ["晚","风","摇","树"]

            # 处理下联（需添加BOS/EOS）
            dec = dec_line.strip().replace(' ', '')
            if not dec:
                print(f"第{line_num}行下联内容为空，已跳过")
                continue
            dec_texts.append(['<BOS>'] + list(dec) + ['<EOS>'])  # 例： ["<BOS>","晨","露"...]

            # 打印前3行处理示例
            if line_num <= 3:
                print(f"样本{line_num}处理结果:")
                print(f"上联原始: {enc_line.strip()}")
                print(f"上联处理: {enc} ➔ {list(enc)}")
                print(f"下联原始: {dec_line.strip()}")
                print(f"下联处理: {dec} ➔ {['<BOS>'] + list(dec) + ['<EOS>']}\n")

        except Exception as e:
            print(f"处理第{line_num}行时报错: {str(e)}")
    print(f"\n最终数据统计:")
    print(f"有效上联数量: {len(enc_texts)}")
    print(f"有效下联数量: {len(dec_texts)}")
    return enc_texts, dec_texts


def get_collate_fn(enc_vocab, dec_vocab):
    def collate_fn(batch):
        enc_batch, dec_batch, target_batch = [], [], []

        for enc_text, dec_text in batch:
            # 处理encoder输入: 转换为索引
            enc_ids = [enc_vocab.stoi.get(c, enc_vocab.stoi[enc_vocab.unk])
                       for c in enc_text]
            # 处理decoder输入和target：
            # decoder_input要加BOS（这里已在数据中添加），target要加EOS
            dec_ids = [dec_vocab.stoi.get(c, dec_vocab.stoi[dec_vocab.unk])
                       for c in dec_text]

            # Encoder输入： [t1, t2,..., tn]
            enc_batch.append(torch.tensor(enc_ids, dtype=torch.long))

            # Decoder输入： [BOS, w1, w2, ..., EOS] → 输入到Decoder的是[BOS, w1, w2,..., w_n]
            dec_batch_ = dec_ids[:-1]  # 去EOS作为输入
            # Target输出： [w1, w2,..., w_n, EOS] （去BOS）
            target_ = dec_ids[1:]  # target比decoder输入右移一位

            dec_batch.append(torch.tensor(dec_batch_, dtype=torch.long))
            target_batch.append(torch.tensor(target_, dtype=torch.long))

        # 处理pad并排序
        enc_padded = pad_sequence(enc_batch,
                                  batch_first=True,
                                  padding_value=enc_vocab.stoi[enc_vocab.pad])
        dec_padded = pad_sequence(dec_batch,
                                  batch_first=True,
                                  padding_value=dec_vocab.stoi[dec_vocab.pad])
        target_padded = pad_sequence(target_batch,
                                     batch_first=True,
                                     padding_value=dec_vocab.stoi[dec_vocab.pad])

        return enc_padded, dec_padded, target_padded

    return collate_fn
