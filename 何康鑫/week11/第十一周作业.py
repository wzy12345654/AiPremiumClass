import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from seqeval.metrics import f1_score, classification_report

# 1. 加载数据集
ds = load_dataset("path/to/msra_ner_k_V3")  

# 2. 数据预处理
def process_data(examples):
    # 提取文本和实体标签
    texts = examples["text"]
    ents = examples["knowledge"]
    
    # 初始化标签列表
    labels = []
    for ent in ents:
        words = ent.split()
        label_ids = [0] * len(words)
        current_entity = None
        for i, word in enumerate(words):
            if word.startswith("B-"):
                current_entity = word[2:]
                label_ids[i] = 1  # B-ENTITY
            elif word.startswith("I-") and current_entity == word[2:]:
                label_ids[i] = 2  # I-ENTITY
            else:
                label_ids[i] = 0
        labels.append(label_ids)
    
    # 将标签转换为Tensor
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, max_length=512)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 初始化Tokenizer和Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=3)  # 3类：O, B-ENTITY, I-ENTITY

# 创建Dataset对象
dataset = Dataset.from_dict({"text": ds["text"], "knowledge": ds["knowledge"]})
dataset = dataset.map(process_data, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# 4. 创建DataCollator
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)

# 5. 创建Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 6. 推理
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    # 解码预测结果
    predicted_labels = predictions[0].numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # 提取实体
    entities = []
    current_entity = []
    current_type = None
    for token, label in zip(tokens, predicted_labels):
        if label == 1:  # B-ENTITY
            if current_entity:
                entities.append({"entity": current_type, "content": "".join(current_entity)})
                current_entity = []
            current_type = "ENTITY"  # 根据实际情况替换为具体实体类型
            current_entity.append(token)
        elif label == 2:  # I-ENTITY
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({"entity": current_type, "content": "".join(current_entity)})
                current_entity = []
            current_type = None
    
    if current_entity:
        entities.append({"entity": current_type, "content": "".join(current_entity)})
    
    return entities

# 测试推理
input_text = "双方确定了今后发展中美关系的指导方针。"
result = predict(input_text)
print(result)
