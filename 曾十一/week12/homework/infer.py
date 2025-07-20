import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

def extract_entities(text, model_dir="/mnt/data_1/zfy/self/homework/12/ner_ddp_output/checkpoint-4221"):
    # === 加载分词器与模型 ===
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # === 编码输入文本 ===
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    # === 获取标签映射表 ===
    id2label = model.config.id2label
    input_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())

    # === 实体抽取逻辑 ===
    entities = []
    current_entity = None

    for token, pred in zip(input_tokens, predictions):
        label = id2label[pred]
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": label[2:], "content": token.replace("##", "")}
        elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
            current_entity["content"] += token.replace("##", "")
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities

# 示例用法
if __name__ == "__main__":
    text = "双方确定了今后发展中美关系的指导方针。"
    extracted = extract_entities(text)
    print(extracted)
