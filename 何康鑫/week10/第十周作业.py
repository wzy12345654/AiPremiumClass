import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TensorBoardCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1. 数据预处理
class JDReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载数据集
df = pd.read_csv("data/JD_Comment_Labels.csv") 
texts = df['comment'].tolist()
labels = df['label'].astype(int).tolist()

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 初始化tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

# 创建数据集实例
train_dataset = JDReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = JDReviewDataset(val_texts, val_labels, tokenizer)

# 2. 模型配置
model_name = "bert-base-chinese"
num_labels = len(df['label'].unique())

# 冻结BERT参数的模型
class FrozenBertModel(AutoModelForSequenceClassification):
    def __init__(self, *args, ​**kwargs):
        super().__init__(*args, ​**kwargs)
        # 冻结BERT所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    report_to="tensorboard"
)

# 3. 训练流程
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "frozen_f1": f1}

# 不冻结参数训练
model_unfrozen = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)
trainer_unfrozen = Trainer(
    model=model_unfrozen,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback("./tb_logs_unfrozen")]
)
trainer_unfrozen.train()

# 冻结参数训练
model_frozen = FrozenBertModel.from_pretrained(
    model_name, num_labels=num_labels
)
trainer_frozen = Trainer(
    model=model_frozen,
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback("./tb_logs_frozen")]
)
trainer_frozen.train()

# 4. 模型保存与预测
best_unfrozen = trainer_unfrozen.best_model
best_unfrozen.save_pretrained("./models/unfrozen_model")
tokenizer.save_pretrained("./models/unfrozen_model")

best_frozen = trainer_frozen.best_model
best_frozen.save_pretrained("./models/frozen_model")
tokenizer.save_pretrained("./models/frozen_model")

# 预测函数
def predict(text, model_path, tokenizer_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
    return predicted_label

# 示例预测
print(predict("这个手机续航太差了！", "./models/unfrozen_model", "./models/unfrozen_model"))
print(predict("物流快服务好！", "./models/frozen_model", "./models/frozen_model"))
