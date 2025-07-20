# 构建encoder和docoder的词典

# 模型训练数据： X：([enc_token_matrix], [dec_token_matrix] shifted right)，
# y [dec_token_matrix] shifted

# 1. 通过词典把token转换为token_index
# 2. 通过Dataloader把encoder，decoder封装为带有batch的训练数据
# 3. Dataloader的collate_fn调用自定义转换方法，填充模型训练数据
#    3.1 encoder矩阵使用pad_sequence填充
#    3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
#    3.3 decoder后面部分训练目标 dec_token_matrix[:,1:,:]
# 4. 创建mask
#    4.1 dec_mask 上三角填充-inf的mask
#    4.2 enc_pad_mask: (enc矩阵 == 0）
#    4.3 dec_pad_mask: (dec矩阵 == 0)
# 5. 创建模型（根据GPU内存大小设计编码和解码器参数和层数）、优化器、损失
# 6. 训练模型并保存

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformer_model import Seq2SeqTransformer

# 1. 构建词典
def build_vocab(token_lists):
    """
    构建词汇表，将输入的 token 列表转换为 token 到索引的映射。

    参数:
    token_lists (list of list): 二维列表，每个子列表包含一系列 token。

    返回:
    dict: 词汇表，键为 token，值为对应的索引。
    """
    # 初始化词汇表，包含特殊 token 及其索引
    vocab = {'<pad>': 0, '<s>': 1, '</s>': 2}
    # 从 3 开始分配索引
    idx = 3
    # 遍历每个 token 列表
    for tokens in token_lists:
        # 遍历当前列表中的每个 token
        for t in tokens:
            # 若 token 不在词汇表中
            if t not in vocab:
                # 将该 token 添加到词汇表并分配索引
                vocab[t] = idx
                # 索引值加 1
                idx += 1
    return vocab

# 2. 数据集
class MyDataset(Dataset):
    def __init__(self, enc_tokens, dec_tokens, enc_vocab, dec_vocab):
        """
        初始化 MyDataset 类的实例。

        参数:
        enc_tokens (list of list): 编码器输入的 token 列表，二维列表，每个子列表代表一个样本的 token 序列。
        dec_tokens (list of list): 解码器输入的 token 列表，二维列表，每个子列表代表一个样本的 token 序列。
        enc_vocab (dict): 编码器使用的词汇表，键为 token，值为对应的索引。
        dec_vocab (dict): 解码器使用的词汇表，键为 token，值为对应的索引。
        """
        # 存储编码器输入的 token 列表
        self.enc_tokens = enc_tokens
        # 存储解码器输入的 token 列表
        self.dec_tokens = dec_tokens
        # 存储编码器使用的词汇表
        self.enc_vocab = enc_vocab
        # 存储解码器使用的词汇表
        self.dec_vocab = dec_vocab

    def __len__(self):
        """
        返回数据集的样本数量。

        由于编码器输入的 token 列表和数据集样本是一一对应的，
        因此通过获取编码器输入的 token 列表的长度，即可得到数据集的样本数量。

        返回:
        int: 数据集的样本数量。
        """
        return len(self.enc_tokens)

    def __getitem__(self, idx):
        """
        根据给定的索引获取数据集中的一个样本，并将样本中的 token 转换为对应的索引。

        参数:
        idx (int): 样本的索引。

        返回:
        tuple: 包含两个 torch.Tensor 对象，分别为编码器输入和解码器输入的 token 索引序列。
        """
        # 根据索引 idx 从编码器输入的 token 列表中取出对应的样本，
        # 并将其中的每个 token 转换为编码器词汇表中对应的索引，存储在列表 enc 中
        enc = [self.enc_vocab[t] for t in self.enc_tokens[idx]]
        # 根据索引 idx 从解码器输入的 token 列表中取出对应的样本，
        # 并将其中的每个 token 转换为解码器词汇表中对应的索引，存储在列表 dec 中
        dec = [self.dec_vocab[t] for t in self.dec_tokens[idx]]
        # 将编码器输入的 token 索引列表转换为 torch.Tensor 对象，数据类型为 torch.long
        # 将解码器输入的 token 索引列表转换为 torch.Tensor 对象，数据类型为 torch.long
        # 最后返回这两个 Tensor 对象组成的元组
        return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)

# 3. collate_fn
def collate_fn(batch):
    """
    自定义的 collate 函数，用于处理从数据集中取出的一批样本，
    对编码器和解码器的输入序列进行填充，并生成解码器的输入和目标输出。

    参数:
    batch (list): 包含一批样本的列表，每个样本是一个元组，包含编码器输入和解码器输入的 token 索引序列。

    返回:
    tuple: 包含三个 torch.Tensor 对象，分别为编码器输入、解码器输入和解码器目标输出。
    """
    # 将一批样本拆分为编码器输入和解码器输入的两个元组
    enc_batch, dec_batch = zip(*batch)
    # 使用 pad_sequence 函数对编码器输入序列进行填充，使其长度一致
    # batch_first=True 表示输出的张量维度为 [batch_size, sequence_length]
    # padding_value=0 表示使用 0 作为填充值
    enc_batch = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    # 对解码器输入序列进行同样的填充操作
    dec_batch = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    # 解码器的输入序列去掉最后一个 token，作为模型训练时的输入
    dec_in = dec_batch[:, :-1]
    # 解码器的目标输出序列去掉第一个 token，作为模型训练时的真实标签
    dec_out = dec_batch[:, 1:]
    return enc_batch, dec_in, dec_out

# 4. mask
def generate_square_subsequent_mask(sz):
    """
    生成一个方形的后续掩码（subsequent mask），用于在序列解码时防止模型看到未来的信息。

    参数:
    sz (int): 掩码矩阵的大小，通常为目标序列的长度。

    返回:
    torch.Tensor: 形状为 (sz, sz) 的掩码矩阵，上三角部分（除对角线）为负无穷，其余部分为 0。
    """
    # 创建一个大小为 (sz, sz) 的全 1 矩阵，将其所有元素乘以负无穷
    # 然后使用 torch.triu 函数提取该矩阵的上三角部分，对角线偏移量为 1（即不包含对角线）
    # 这样得到的矩阵上三角部分（除对角线）为负无穷，其余部分为 0
    mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
    return mask

# ========== 主流程 ==========
if __name__ == '__main__':
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "风弦未拨心先乱，夜幕已沉梦更闲"
    chs = list(corpus)
    
    enc_tokens, dec_tokens = [],[]

    for i in range(1,len(chs)):
        enc = chs[:i]
        dec = ['<s>'] + chs[i:] + ['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    # 构建词典
    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)
    inv_dec_vocab = {v: k for k, v in dec_vocab.items()}

    # 构建数据集和dataloader
    dataset = MyDataset(enc_tokens, dec_tokens, enc_vocab, dec_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 模型参数
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead, num_enc_layers, num_dec_layers, dim_forward, dropout, enc_voc_size, dec_voc_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    # 训练
    for epoch in range(50):
        model.train()
        total_loss = 0
        for enc_batch, dec_in, dec_out in dataloader:
            enc_batch, dec_in, dec_out = enc_batch.to(device), dec_in.to(device), dec_out.to(device)
            tgt_mask = generate_square_subsequent_mask(dec_in.size(1)).to(device)
            enc_pad_mask = (enc_batch == 0)
            dec_pad_mask = (dec_in == 0)
            logits = model(enc_batch, dec_in, tgt_mask, enc_pad_mask, dec_pad_mask)
            # logits: [B, T, V]  dec_out: [B, T]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer.pth')
    print("模型已保存。")