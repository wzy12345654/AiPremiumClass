import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# 使用中文对联数据集训练带有attention的seq2seq模型，利用tensorboard跟踪。
# https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets

# 基本设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 20
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 64
EPOCHS = 1

# 读取字典
class Lang:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

# 数据处理
def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def prepare_data(train_in_path, train_out_path, test_in_path=None, test_out_path=None):
    train_input_lines = read_lines(train_in_path)
    train_output_lines = read_lines(train_out_path)
    train_pairs = list(zip(train_input_lines, train_output_lines))

    input_lang = Lang()
    output_lang = Lang()

    for in_sent, out_sent in train_pairs:
        input_lang.add_sentence(in_sent)
        output_lang.add_sentence(out_sent)
    
    val_pairs = []
    if test_in_path and test_out_path:
        val_input_lines = read_lines(test_in_path)
        val_output_lines = read_lines(test_out_path)
        val_pairs = list(zip(val_input_lines, val_output_lines))
        # Ensure validation words are in lang, even if not used for training lang growth
        for in_sent, out_sent in val_pairs:
            input_lang.add_sentence(in_sent) # Add to lang to prevent unknown words
            output_lang.add_sentence(out_sent)

    return input_lang, output_lang, train_pairs, val_pairs

def sentence_to_tensor(lang, sentence, max_len):
    idxs = [lang.word2idx[c] for c in sentence]
    idxs = [lang.word2idx['<SOS>']] + idxs + [lang.word2idx['<EOS>']]
    if len(idxs) < max_len:
        idxs += [lang.word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return torch.tensor(idxs, dtype=torch.long)

# 自定义数据集
class CoupletDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang, max_len):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        in_sent, out_sent = self.pairs[idx]
        x = sentence_to_tensor(self.input_lang, in_sent, self.max_len)
        y = sentence_to_tensor(self.output_lang, out_sent, self.max_len)
        return x, y

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, B, H] -> [B, 1, H] -> repeat -> [B, T, H]
        hidden = hidden.squeeze(0).unsqueeze(1)  # [B, 1, H]
        T = encoder_outputs.size(1)
        hidden = hidden.repeat(1, T, 1)          # [B, T, H]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        
        # 修复：正确扩展v的维度以匹配批处理大小
        batch_size = encoder_outputs.size(0)
        v = self.v.repeat(batch_size, 1).unsqueeze(2)  # [B, H, 1]
        
        energy = torch.bmm(energy, v)  # [B, T, 1]
        energy = energy.squeeze(2)     # [B, T]
        attn_weights = torch.softmax(energy, dim=1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]
        return context

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attn = Attention(hidden_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        emb = self.embedding(x).unsqueeze(1)  # [B, 1, E]
        context = self.attn(hidden, encoder_outputs)  # [B, 1, H]
        rnn_input = torch.cat((emb, context), dim=2)  # [B, 1, E+H]
        out, hidden = self.rnn(rnn_input, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden

# 训练
def train():
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/train/in.txt',
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/train/out.txt',
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/test/in.txt',  # Validation input
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/test/out.txt' # Validation output
    )
    
    # 创建数据集和数据加载器
    train_dataset = CoupletDataset(train_pairs, input_lang, output_lang, MAX_LENGTH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = CoupletDataset(test_pairs, input_lang, output_lang, MAX_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    encoder = Encoder(input_lang.n_words, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    decoder = Decoder(output_lang.n_words, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Training]")
        for batch_idx, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            
            # 将数据移动到设备上
            x = x.to(DEVICE)  # [B, T]
            y = y.to(DEVICE)  # [B, T]
            
            batch_size = x.size(0)
            
            # 编码器前向传播
            encoder_outputs, hidden = encoder(x)
            
            # 解码器输入从SOS开始
            dec_input = y[:, 0]  # 获取每个序列的第一个token (SOS)
            loss = 0

            # 逐步解码
            for t in range(1, y.size(1)):
                output, hidden = decoder(dec_input, hidden, encoder_outputs)
                loss += loss_fn(output, y[:, t])
                dec_input = y[:, t]  # 使用教师强制

            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算平均损失
            avg_loss = loss.item() / (y.size(1) - 1)  # 减1是因为我们不预测SOS
            total_loss += avg_loss
            
            # 更新进度条
            pbar.set_description(f"Epoch {epoch+1} [Training], Loss: {avg_loss:.4f}")

        # 记录每个epoch的平均训练损失
        epoch_train_loss = total_loss / len(train_dataloader)
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        print(f"Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f}")

        # 验证步骤
        encoder.eval()
        decoder.eval()
        total_val_loss = 0
        val_pbar = tqdm(test_dataloader, desc=f"Epoch {epoch+1} [Validation]")
        with torch.no_grad():
            for x_val, y_val in val_pbar:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)
                
                encoder_outputs_val, hidden_val = encoder(x_val)
                dec_input_val = y_val[:, 0]
                loss_val = 0
                for t in range(1, y_val.size(1)):
                    output_val, hidden_val = decoder(dec_input_val, hidden_val, encoder_outputs_val)
                    loss_val += loss_fn(output_val, y_val[:, t])
                    dec_input_val = y_val[:, t] # Teacher forcing for validation too for simplicity
                
                avg_val_loss = loss_val.item() / (y_val.size(1) - 1)
                total_val_loss += avg_val_loss
                val_pbar.set_description(f"Epoch {epoch+1} [Validation], Val Loss: {avg_val_loss:.4f}")

        epoch_val_loss = total_val_loss / len(test_dataloader)
        writer.add_scalar("Loss/validation", epoch_val_loss, epoch)
        print(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}")
        
        # 每一轮训练后保存模型
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(encoder.state_dict(), f"checkpoints/encoder_epoch{epoch+1}.pt")
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch{epoch+1}.pt")
        
    writer.close()

# 推理函数
def inference(encoder, decoder, input_sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = sentence_to_tensor(input_lang, input_sentence, max_length).unsqueeze(0).to(DEVICE)
        encoder_outputs, hidden = encoder(input_tensor)

        # 解码器初始输入为 <SOS> 
        # dec_input 的形状应为 [1] (batch_size=1)
        dec_input = torch.tensor([output_lang.word2idx['<SOS>']], device=DEVICE) # 形状 [1]
        decoded_words = []

        for _ in range(max_length):
            output, hidden = decoder(dec_input, hidden, encoder_outputs)
            topv, topi = output.data.topk(1) # output.data.topk(1) or output.topk(1)
            
            if topi.item() == output_lang.word2idx['<EOS>']:
                # decoded_words.append('<EOS>') # Optional: if you want to see EOS in output
                break
            else:
                decoded_words.append(output_lang.idx2word[topi.item()])

            dec_input = topi.squeeze().detach() # 使用模型自身的预测作为下一个输入
            if dec_input.dim() == 0: # Ensure it's a 1D tensor for unsqueeze later if needed by embedding
                dec_input = dec_input.unsqueeze(0)

        return ''.join(decoded_words)

if __name__ == '__main__':
    # 首先进行训练
    # train()

    # 训练完成后，加载模型并进行推理
    print("\nStarting inference after training...")
    # 准备语言对象 (需要从训练数据中构建或加载)
    # 为了确保词典一致，推理时也应该加载包含验证集词汇的词典
    input_lang_inf, output_lang_inf, _, _ = prepare_data(
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/train/in.txt',
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/train/out.txt',
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/test/in.txt', 
        '/Users/chenxing/AI/AiPremiumClass/陈兴/week08/data/couplet/test/out.txt'
    )

    # 加载最后一次保存的模型
    # 假设 EPOCHS = 1, 则模型文件为 encoder_epoch1.pt 和 decoder_epoch1.pt
    # 可能需要根据实际的 EPOCHS 调整文件名
    encoder_path = f"/Users/chenxing/AI/AiPremiumClass/陈兴/week08/checkpoints/encoder_epoch{EPOCHS}.pt"
    decoder_path = f"/Users/chenxing/AI/AiPremiumClass/陈兴/week08/checkpoints/decoder_epoch{EPOCHS}.pt"

    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        encoder_inf = Encoder(input_lang_inf.n_words, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        decoder_inf = Decoder(output_lang_inf.n_words, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)

        encoder_inf.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=True))
        decoder_inf.load_state_dict(torch.load(decoder_path, map_location=DEVICE, weights_only=True))

        # 测试一些上联
        test_couplets = [
            "海内存知己",
            "春风又绿江南岸",
            "一行白鹭上青天"
        ]

        for couplet_in in test_couplets:
            generated_couplet = inference(encoder_inf, decoder_inf, couplet_in, input_lang_inf, output_lang_inf)
            print(f"上联: {couplet_in}")
            print(f"下联: {generated_couplet}")
            print("---")
    else:
        print(f"Model checkpoints not found at {encoder_path} or {decoder_path}. Skipping inference.")
