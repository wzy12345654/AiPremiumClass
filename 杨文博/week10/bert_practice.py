import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1. 加载数据
df = pd.read_excel('/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx', engine='openpyxl')
texts = df['评价内容(content)'].tolist()  # 假设列名为 'comment'
scores = df['评分（总分5分）(score)'].tolist()  # 假设列名为 'score'

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, scores, test_size=0.2, random_state=42
)

# 2. 加载分词器和模型
model_name = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=5  # 假设分数是1~5的整数（分类任务）
)


# 3. 自定义Dataset类
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx]) - 1  # 假设分数是1~5，调整为0~4的类别

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 创建Dataset和DataLoader
train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
val_dataset = CommentDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 初始化TensorBoard
writer = SummaryWriter('runs/comment_classification_experiment')

# 5. 训练循环
num_epochs = 3
global_step = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # 记录训练损失
        writer.add_scalar('training_loss', loss.item(), global_step)
        global_step += 1

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # 验证集评估
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            val_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    # 记录验证指标
    writer.add_scalar('val_loss', avg_val_loss, epoch)
    writer.add_scalar('val_accuracy', val_accuracy, epoch)

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# 关闭TensorBoard写入器
writer.close()