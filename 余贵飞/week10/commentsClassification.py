
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tensorboard import SummaryWriter

# 1. 数据预处理类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels  
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 模型训练函数
def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    # 设备检查使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 添加tensorbroad 数据记录
    writer = SummaryWriter()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training loss', loss.item(), epoch)
        avg_train_loss = total_loss / len(train_loader)
        # 模型评估
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train loss: {avg_train_loss:.4f}, Val acc: {val_acc:.4f}')
        # 模型精度记录
        writer.add_scalar('Validation accuracy', val_acc, epoch)
        # 模型保存
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# 3. 模型评估函数
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)
    return correct.double() / total
