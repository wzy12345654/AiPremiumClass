import pandas as pd
import jieba
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import matplotlib.pyplot as plt

# 配置参数
BATCH_SIZE = 32
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_CLASSES = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # 过滤评分并转换标签
    df = df[(df['rating'] <= 2) | (df['rating'] >= 4)]
    df['label'] = df['rating'].apply(lambda x: 1 if x <= 2 else 0)
    return df[['comment', 'label']].values.tolist()

def clean_text(text):
    # 清理HTML标签、特殊符号和数字
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip().lower()

# 分词工具接口
class Tokenizer:
    def tokenize(self, text):
        raise NotImplementedError
    
class JiebaTokenizer(Tokenizer):
    def tokenize(self, text):
        return list(jieba.cut(clean_text(text), cut_all=False))
    
class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def tokenize(self, text):
        return self.sp.encode_as_pieces(clean_text(text))

# 构建词汇表
def build_vocab(tokenized_data, min_freq=5):
    counter = Counter()
    for tokens in tokenized_data:
        counter.update(tokens)
    vocab = ['[PAD]', '[UNK]'] + [token for token, cnt in counter.items() if cnt >= min_freq]
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    return vocab, vocab_dict

# 自定义数据集
class MovieReviewDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, max_len):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer.tokenize(text)
        # 转换为索引并填充
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        token_ids = token_ids[:self.max_len] + [self.vocab['[PAD]']] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids), torch.tensor(label, dtype=torch.float)

# 构建模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# 数据加载流程
def load_and_preprocess_data(csv_path, tokenizer, max_len, test_size=0.2):
    # 加载原始数据
    data = load_data(csv_path)
    
    # 划分训练验证集
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # 分别构建词汇表
    train_tokenized = [tokenizer.tokenize(text) for text, _ in train_data]
    vocab, vocab_dict = build_vocab(train_tokenized)
    
    # 创建数据集和DataLoader
    train_dataset = MovieReviewDataset(train_data, vocab, tokenizer, max_len)
    val_dataset = MovieReviewDataset(val_data, vocab, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, len(vocab), vocab_dict

# 训练函数
def train_model(model, train_loader, val_loader, vocab_size, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()  # 二分类任务使用BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses, val_accuracies = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).sigmoid().round()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_accuracy = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
        val_losses.append(criterion(model(inputs).squeeze(), labels.float()).item())
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).sigmoid().round()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return accuracy


if __name__ == "__main__":
    # 参数设置
    CSV_PATH = "douban_comments.csv"
    MODEL_SAVE_PATH = "text_classifier.pth"
    VOCAB_SAVE_PATH = "vocab.pkl"
    
    # 1. 数据预处理 & 加载
    print("正在加载数据...")
    train_loader, val_loader, vocab_size, vocab_dict = load_and_preprocess_data(
        csv_path=CSV_PATH,
        tokenizer=JiebaTokenizer(),
        max_len=MAX_SEQ_LEN,
        test_size=0.2
    )
    
    # 2. 训练模型（使用jieba分词）
    print("\n开始训练（Jieba分词）...")
    model_jieba = TextClassifier(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, 
                                 hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
    train_losses_jieba, val_losses_jieba, val_accs_jieba = train_model(
        model=model_jieba,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )
    
    # 3. 保存模型和词汇表
    torch.save(model_jieba.state_dict(), MODEL_SAVE_PATH)
    import pickle
    with open(VOCAB_SAVE_PATH, 'wb') as f:
        pickle.dump(vocab_dict, f)
    
    # 4. 测试不同分词工具
    print("\n切换至SentencePiece分词器...")
    # 训练SentencePiece模型
    spm.SentencePieceTrainer.train(
        input=CSV_PATH,
        model_prefix="sp_model",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe'
    )
    tokenizer_sp = SentencePieceTokenizer(model_path="sp_model.model")
    
    # 重新加载数据
    train_loader_sp, val_loader_sp, _, _ = load_and_preprocess_data(
        csv_path=CSV_PATH,
        tokenizer=tokenizer_sp,
        max_len=MAX_SEQ_LEN,
        test_size=0.2
    )
    
    # 训练模型（SentencePiece分词）
    model_sp = TextClassifier(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, 
                              hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
    train_losses_sp, val_losses_sp, val_accs_sp = train_model(
        model=model_sp,
        train_loader=train_loader_sp,
        val_loader=val_loader_sp,
        vocab_size=vocab_size,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )
    
    # 5. 结果对比
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_jieba, label='Jieba Train Loss')
    plt.plot(val_losses_jieba, label='Jieba Val Loss')
    plt.plot(train_losses_sp, label='SP Train Loss')
    plt.plot(val_losses_sp, label='SP Val Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs_jieba, label='Jieba Val Acc')
    plt.plot(val_accs_sp, label='SP Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


class AdvancedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(768, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        bert_output = self.bert(embedded)[1]  # 使用[CLS] token
        return self.fc(bert_output)

  from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).sigmoid().round()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
