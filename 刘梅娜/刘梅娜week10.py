from transformers import BertModel
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer
from datasets import Dataset
import torch
import numpy as np # linear algebra
import pandas as pd
import transformers


class CustomBertModel(torch.nn.Module):
    # 定义一个自定义的Bert模型类
    def __init__(self, num_labels=3):
        # 初始化函数，num_labels为分类标签的数量，默认为3
        super(CustomBertModel, self).__init__()
        # 调用父类的初始化函数
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 从预训练的Bert模型中加载中文模型
        self.classifier = torch.nn.Linear(768, num_labels)

        # 定义一个线性分类器，输入维度为768，输出维度为num_labels
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 定义前向传播函数
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 调用Bert模型，输入为input_ids和attention_mask
        sequence_output = outputs[1]
        # 获取Bert模型的输出
        logits = self.classifier(sequence_output)
        
        # 将Bert模型的输出通过线性分类器得到logits
        if labels is not None:
            # 如果有标签
            loss_fct = torch.nn.CrossEntropyLoss()
            # 定义交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 计算损失
            return {"loss": loss, "logits": logits}
        else:
            # 如果没有标签
            return {"logits": logits}

## 3. TensorBoard集成ß

## 4. 模型训练与评估
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",# 每个 epoch 结束时进行评估
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to='tensorboard'
)


## 5. 模型保存与预测

def predict(text):
    # 加载预训练的BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 对输入文本进行分词，并添加padding和truncation操作，最大长度为512，返回PyTorch张量
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # 不计算梯度，直接进行前向传播
    with torch.no_grad():
        # 使用模型进行预测，返回输出
        outputs = model(**inputs)
        # 对输出进行softmax操作，得到概率分布
        probs = torch.softmax(outputs.logits, dim=1)
        # 取概率最大的标签
        pred_label = torch.argmax(probs, dim=1).item()
    
    # 返回预测标签和概率分布
    return pred_label, probs.squeeze().tolist()

# 示例用法
# text = "这是一段职位描述..."
# predict(text)

# 读取 Excel 文件
file_path = 'jd_comment_data.xlsx'  # 修改为你的文件路径
df = pd.read_excel(file_path)

# 显示前几行数据以确认读取正确
#print(df.head())

# 假设 '职位描述' 是你要进行预测的列名
descriptions = df['评价内容(content)'].tolist()
dataset = Dataset.from_pandas(df[['评价内容(content)', '标签']])

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize_function(examples):
    return tokenizer(examples["评价内容(content)"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 划分训练集和验证集（例如 80% 训练，20% 验证）
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["test"]

model = CustomBertModel(num_labels=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

