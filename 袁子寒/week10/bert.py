import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


df = pd.read_excel('jd_comment_data.xlsx')
# print(df.columns)
filterd= df['评价内容(content)'] != "此用户未填写评价内容"
data_df = df[filterd][['评价内容(content)','评分（总分5分）(score)']]
# print(data_df.head())
# 定义数据集类
class JDCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 定义BERT分类模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, model_name):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 添加训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条信息
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条信息
                val_pbar.set_postfix({
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_acc = 100 * val_correct / val_total
        
        # 记录到tensorboard
        writer.add_scalar(f'{model_name}/train_loss', avg_loss, epoch)
        writer.add_scalar(f'{model_name}/train_acc', train_acc, epoch)
        writer.add_scalar(f'{model_name}/val_acc', val_acc, epoch)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pth')

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer和预训练模型
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    bert_model = AutoModel.from_pretrained('bert-base-chinese')
    
    # 准备数据
    texts = data_df['评价内容(content)'].values
    labels = data_df['评分（总分5分）(score)'].values - 1  # 转换为0-4的标签
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据集
    train_dataset = JDCommentDataset(train_texts, train_labels, tokenizer)
    val_dataset = JDCommentDataset(val_texts, val_labels, tokenizer)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 设置tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs/jd_comment_classification')
    
    # 训练冻结BERT的模型
    frozen_model = BERTClassifier(bert_model, num_classes=5, freeze_bert=True)
    frozen_model = frozen_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(frozen_model.parameters(), lr=2e-5)
    
    print("Training frozen BERT model...")
    train_model(frozen_model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=5, device=device, writer=writer, model_name='frozen_bert')
    
    # 训练不冻结BERT的模型
    unfrozen_model = BERTClassifier(bert_model, num_classes=5, freeze_bert=False)
    unfrozen_model = unfrozen_model.to(device)
    optimizer = optim.AdamW(unfrozen_model.parameters(), lr=2e-5)
    
    print("\nTraining unfrozen BERT model...")
    train_model(unfrozen_model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=5, device=device, writer=writer, model_name='unfrozen_bert')
    
    writer.close()

# 预测函数
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item() + 1  # 转换回1-5的评分

if __name__ == '__main__':
    # main()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = AutoModel.from_pretrained('bert-base-chinese')
    # frozen_model = BERTClassifier(bert_model, num_classes=5, freeze_bert=True)
    # frozen_model = frozen_model.to(device)
    # frozen_model.load_state_dict(torch.load('best_frozen_bert.pth'))
    # result = predict_sentiment("商品质量很好，物流也很快，服务态度很好，下次还会购买", frozen_model, tokenizer, device)
    # print(f"预测结果: {result}")
    unfrozen_model = BERTClassifier(bert_model, num_classes=5, freeze_bert=False)
    unfrozen_model = unfrozen_model.to(device)
    unfrozen_model.load_state_dict(torch.load('best_unfrozen_bert.pth'))
    result = predict_sentiment("商品质量很好，物流也很快，服务态度很好，下次还会购买", unfrozen_model, tokenizer, device)
    print(f"预测结果: {result}")





