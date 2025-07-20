"""
1.根据data/jd_comment_result.xlsx提供的 JD评论语料，用bert模型进行文本分类训练
2.调整模型训练参数，添加tensorboard跟踪，对比bert冻结和不冻结之间的训练差异。
"""
# 导入必要的库
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 定义数据集类
class JDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])  # 确保文本是字符串类型
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,  # 输入文本
            add_special_tokens=True,  # 添加特殊标记 [CLS] 和 [SEP]
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断到最大长度
            max_length=self.max_length,  # 最大长度
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }    

def load_data(file_path):
    df = pd.read_excel(file_path)
    texts = df['评价内容'].tolist()
    labels = df['评价类型'].tolist()
    return texts, labels

def train_model(model, train_dataloader, optimizer, device, writer, epoch, is_frozen):
    model.train()
    total_loss = 0
    train_loss_cnt = 0
    tpbar = tqdm(train_dataloader)
    for batch in tpbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        tpbar.set_description(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
        writer.add_scalar(f'Loss/train_{"frozen" if is_frozen else "unfrozen"}', loss.item(), train_loss_cnt)
        train_loss_cnt += 1
    return total_loss / len(train_dataloader)


def evaluate_model(model, val_dataloader, device, writer, epoch, is_frozen):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    writer.add_scalar(f'Accuracy/val_{"frozen" if is_frozen else "unfrozen"}', accuracy, epoch)
    return accuracy

def freeze_bert(model):
    for param in model.bert.parameters():
        param.requires_grad = False


def unfreeze_bert(model):
    for param in model.bert.parameters():
        param.requires_grad = True

if __name__ == '__main__':
    # 配置参数
    file_path = 'data/jd_comment_result_100.xlsx'
    model_name = 'bert-base-chinese'
    max_length = 128
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    texts, labels = load_data(file_path)

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 加载预训练模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类任务
    model.to(device)

    # 创建数据集实例
    train_dataset = JDDataset(texts, labels, tokenizer, max_length)
    #val_dataset = JDDataset(val_texts, val_labels, tokenizer, max_length)
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练模型
    writer = SummaryWriter()

    # 训练冻结的 BERT 模型
    frozen_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    freeze_bert(frozen_model)
    optimizer_frozen = AdamW(frozen_model.parameters(), lr=learning_rate)
    print("Training frozen BERT model...")
    for epoch in range(epochs):
        train_loss = train_model(frozen_model, train_dataloader, optimizer_frozen, device, writer, epoch, True)
        print(f'Frozen - Train Loss: {train_loss:.4f}')
        #val_accuracy = evaluate_model(frozen_model, val_dataloader, device, writer, epoch, True)
        #print(f'Frozen - Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
    frozen_model.save_pretrained('data/bert_comment_classifier_frozen')
    tokenizer.save_pretrained('data/bert_comment_classifier_frozen')

    # 训练未冻结的 BERT 模型
    unfrozen_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    unfreeze_bert(unfrozen_model)
    optimizer_unfrozen = AdamW(unfrozen_model.parameters(), lr=learning_rate)
    print("Training unfrozen BERT model...")
    for epoch in range(epochs):
        train_loss = train_model(unfrozen_model, train_dataloader, optimizer_unfrozen, device, writer, epoch, False)
        print(f'Unfrozen - Train Loss: {train_loss:.4f}')
        #val_accuracy = evaluate_model(unfrozen_model, val_dataloader, device, writer, epoch, False)
        #print(f'Unfrozen - Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}')
    unfrozen_model.save_pretrained('data/bert_comment_classifier_unfrozen')
    tokenizer.save_pretrained('data/bert_comment_classifier_unfrozen')

    writer.close() 