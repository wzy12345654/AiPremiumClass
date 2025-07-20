# 1. 根据提供的kaggle JD评论语料进行文本分类训练
# https://www.kaggle.com/datasets/dosonleung/jd_comment_with_label
# 2. 调整模型训练参数，添加tensorboard跟踪，对比bert冻结和不冻结之间的训练差异。
# 3. 保存模型进行分类预测。
# 京东评论文本分类 - Kaggle版本

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import json
from tqdm.notebook import tqdm  # Kaggle支持notebook进度条

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集路径 - Kaggle环境
DATA_PATH = "/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx"

# 数据预处理
class JDCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    # 在JDCommentDataset类中添加文本预处理
    def preprocess_text(self, text):
        # 基本清洗
        text = text.strip()
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # 应用预处理
        text = self.preprocess_text(text)
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, 
                                  add_special_tokens=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 模型定义
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)  # 增加dropout率
        # 添加更复杂的分类头
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )
        
        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        # 获取BERT的最后一层隐藏状态，而不是只用pooler_output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的最后隐藏状态
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(last_hidden_state)
        logits = self.classifier(x)
        return logits

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, writer, model_name):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        # 使用tqdm显示进度条
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            
            # 更新进度条
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if scheduler:
            scheduler.step()
            
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                
                # 更新进度条
                val_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算验证指标
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # 记录到TensorBoard
        writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{model_name}/train_acc', train_acc, epoch)
        writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
        writer.add_scalar(f'{model_name}/val_acc', val_acc, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 在Kaggle中保存到/kaggle/working/目录
            torch.save(model.state_dict(), f'/kaggle/working/{model_name}_best.pt')
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    return best_val_acc

# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    test_preds, test_labels = [], []
    
    test_progress = tqdm(test_loader, desc='Evaluating')
    with torch.no_grad():
        for batch in test_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(labels.cpu().tolist())
    
    # 计算评估指标
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds)
    
    return accuracy, report

# 预测函数
def predict(text, model, tokenizer, max_length=128):
    model.eval()
    # 预处理文本
    text = text.strip()
    text = ' '.join(text.split())
    
    encoding = tokenizer(text, 
                         add_special_tokens=True,
                         max_length=max_length,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
    
    # 返回预测标签和置信度
    return preds.item(), confidence.item()

# 在文件顶部添加计算类权重的函数
def calculate_class_weights(labels):
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

# 加载本地模型并进行预测的函数
def load_model_and_predict(model_path, sample_texts):
    print(f"\n加载本地模型 {model_path} 进行预测...")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
     
    # 加载保存的模型信息 - 设置weights_only=False解决PyTorch 2.6的兼容性问题
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    model_type = checkpoint['model_type']
    accuracy = checkpoint['accuracy']
    
    # 初始化BERT模型
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    
    # 创建分类器模型实例
    model = BERTClassifier(
        bert_model, 
        num_classes, 
        freeze_bert=(model_type == "冻结BERT")
    ).to(device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"模型类型: {model_type}")
    print(f"模型准确率: {accuracy:.4f}")
    
    # 进行预测
    print("\n示例预测:")
    for text in sample_texts:
        pred, confidence = predict(text, model, tokenizer)
        sentiment = ["负面", "中性", "正面"][pred]  # 现在pred是整数
        print(f"文本: {text}\n预测标签: {pred} ({sentiment}), 置信度: {confidence:.4f}")

# 主函数
def main():
    # 加载数据
    print("加载数据...")
    try:
        # 使用read_excel读取数据
        df = pd.read_excel(DATA_PATH)
        print(f"数据集形状: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 检查数据列名
    print(f"数据列名: {df.columns.tolist()}")
    
    # 根据实际Excel文件的列名调整
    # 评论内容在第3列，评分在第5列
    text_col = df.columns[2]  # 评价内容(content)
    label_col = df.columns[4]  # 评价分(star)
    
    # 数据分割
    texts = df[text_col].values
    
    # 处理标签：将评分转换为分类标签
    # 假设评分是1-5的数值，我们可以将其转换为分类标签
    # 例如：1-2分为负面(0)，3分为中性(1)，4-5分为正面(2)
    def convert_rating_to_label(rating):
        try:
            rating = float(rating)
            if rating <= 2:
                return 0  # 负面
            elif rating < 4:  # 修改为3分到4分之间为中性
                return 1  # 中性
            else:
                return 2  # 正面
        except:
            # 如果无法转换为数值，默认为中性
            return 1
    
    labels = np.array([convert_rating_to_label(rating) for rating in df[label_col].values])
    
    # 查看标签分布
    print(f"标签分布:\n{pd.Series(labels).value_counts()}")
    
    # 分割数据集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 计算类别权重 - 移到这里，确保train_labels已定义
    class_weights = calculate_class_weights(train_labels)
    class_weights = class_weights.to(device)
    
    # 加载tokenizer和预训练模型
    print("加载BERT模型和tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    
    # 创建数据集和数据加载器
    train_dataset = JDCommentDataset(train_texts, train_labels, tokenizer)
    val_dataset = JDCommentDataset(val_texts, val_labels, tokenizer)
    test_dataset = JDCommentDataset(test_texts, test_labels, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 设置TensorBoard - Kaggle环境
    writer = SummaryWriter('/kaggle/working/runs/jd_comments_classification')
    
    # 训练参数
    num_classes = len(set(labels))
    num_epochs = 2  # 增加训练轮数
    learning_rate = 3e-5  # 微调学习率
    weight_decay = 0.01  # 添加权重衰减
    
    # 创建并训练冻结BERT的模型
    print("\n训练冻结BERT的模型...")
    frozen_model = BERTClassifier(bert_model, num_classes, freeze_bert=True).to(device)
    frozen_optimizer = optim.AdamW(frozen_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    frozen_scheduler = optim.lr_scheduler.StepLR(frozen_optimizer, step_size=1, gamma=0.9)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 使用加权损失函数
    
    frozen_best_acc = train_model(
        frozen_model, train_loader, val_loader, frozen_optimizer, criterion, 
        frozen_scheduler, num_epochs, writer, 'frozen_bert'
    )
    
    # 创建并训练非冻结BERT的模型
    print("\n训练非冻结BERT的模型...")
    unfrozen_model = BERTClassifier(BertModel.from_pretrained('bert-base-chinese'), num_classes, freeze_bert=False).to(device)
    unfrozen_optimizer = optim.AdamW(unfrozen_model.parameters(), lr=learning_rate)
    unfrozen_scheduler = optim.lr_scheduler.StepLR(unfrozen_optimizer, step_size=1, gamma=0.9)
    
    unfrozen_best_acc = train_model(
        unfrozen_model, train_loader, val_loader, unfrozen_optimizer, criterion, 
        unfrozen_scheduler, num_epochs, writer, 'unfrozen_bert'
    )
    
    # 加载最佳模型进行评估
    print("\n评估最佳模型...")
    best_model_path = '/kaggle/working/frozen_bert_best.pt' if frozen_best_acc > unfrozen_best_acc else '/kaggle/working/unfrozen_bert_best.pt'
    best_model_type = "冻结BERT" if frozen_best_acc > unfrozen_best_acc else "非冻结BERT"
    
    best_model = BERTClassifier(
        BertModel.from_pretrained('bert-base-chinese'), 
        num_classes, 
        freeze_bert=(best_model_type == "冻结BERT")
    ).to(device)
    
    best_model.load_state_dict(torch.load(best_model_path))
    accuracy, report = evaluate_model(best_model, test_loader)
    
    print(f"最佳模型类型: {best_model_type}")
    print(f"测试准确率: {accuracy:.4f}")
    print(f"分类报告:\n{report}")
    
    # 保存最终模型 - Kaggle环境
    final_model_path = '/kaggle/working/jd_comment_classifier_final.pt'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_type': best_model_type,
        'num_classes': num_classes,
        'accuracy': accuracy
    }, final_model_path)
    
    print(f"最终模型已保存到 {final_model_path}")
    
    # 示例预测
    sample_texts = [
        "这个产品质量很好，我很满意",
        "太差了，完全不值这个价格",
        "一般般吧，没有想象中那么好"
    ]
    
    # 使用训练好的模型直接预测
    print("\n使用当前训练的模型进行预测:")
    for text in sample_texts:
        pred, confidence = predict(text, best_model, tokenizer)  # 正确解包返回值
        sentiment = ["负面", "中性", "正面"][pred]  # 现在pred是整数
        print(f"文本: {text}\n预测标签: {pred} ({sentiment}), 置信度: {confidence:.4f}")
    
    # 加载本地模型进行预测
    load_model_and_predict(final_model_path, sample_texts)
    
    # 关闭TensorBoard writer
    writer.close()

if __name__ == "__main__":
    # main()
    sample_texts = [
        "这个产品质量很好，我很满意",
        "太差了，完全不值这个价格",
        "一般般吧，没有想象中那么好"
    ]
    load_model_and_predict('/Users/chenxing/Downloads/frozen_bert_best.pt', sample_texts)