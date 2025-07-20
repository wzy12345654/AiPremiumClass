from transformers import AutoModelForTokenClassification,AutoTokenizer
import torch
from transformers import TrainingArguments,Trainer
import numpy as np
from datasets import load_dataset


ds = load_dataset("doushabao4766/msra_ner_k_V3")
# 加载模型和tokenizer
model_name = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)

label_list = ds["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
print("标签列表:", label_list)


# 数据预处理
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 特殊token设为-100，计算loss时会忽略
            if word_idx is None:
                label_ids.append(-100)
            # 为token设置对应标签
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # 对于同一个词的分片token，可以设为-100或相同标签
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 应用预处理
tokenized_datasets = ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=ds["train"].column_names
)

# 训练
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./ner_results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
)

from transformers import DataCollatorForTokenClassification

# 初始化数据收集器
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,  # 替换tokenizer参数
    # compute_metrics=compute_metrics
)

trainer.train()

# 3. 保存最终模型
trainer.save_model("./ner_final_model")  # 模型保存路径
tokenizer.save_pretrained("./ner_final_model")

# 6. 推理函数
class NERPredictor:
    def __init__(self, model_path=None):
        if model_path:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.model.eval()

    def predict(self, text):
        # 中文需要先分词（简单按字符分）
        tokens = list(text)

        # Tokenize输入
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            is_split_into_words=True
        )

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测标签
        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_tags = [tags[i] for i in predictions[0].tolist()]

        # 对齐token和标签
        word_ids = inputs.word_ids()

        # 提取实体
        entities = []
        current_entity = None

        for token, tag, word_id in zip(tokens, predicted_tags, word_ids):
            if word_id is None:  # 跳过特殊token
                continue

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'entity': tag[2:],
                    'start': word_id,
                    'end': word_id + 1,
                    'text': tokens[word_id]
                }
            elif tag.startswith('I-') and current_entity:
                if tag[2:] == current_entity['entity']:
                    current_entity['end'] = word_id + 1
                    current_entity['text'] = ''.join(tokens[current_entity['start']:current_entity['end']])
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities


# 7. 主程序
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Text to predict entities')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.predict:
        predictor = NERPredictor("msra_ner_model")
        entities = predictor.predict(args.predict)
        print(f"Text: {args.predict}")
        print("Entities:")
        for ent in entities:
            print(f"{ent['entity']}: {ent['text']} (position: {ent['start']}-{ent['end']})")
    else:
        print("Please specify either --train to train the model or --predict <text> to predict entities")