import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
id2tag = {i: tag for i, tag in enumerate(labels)}
model = AutoModelForTokenClassification.from_pretrained("./ner_model")
tokenizer = AutoTokenizer.from_pretrained("./ner_model")
def extract_entities(text, predictions):
    entities = []
    current_entity = None
    current_type = ""
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    word_ids = tokenizer(text).word_ids()
    predicted_labels = predictions[0].tolist() if isinstance(predictions[0], torch.Tensor) else predictions[0]

    previous_word_idx = None
    current_entity = None
    current_type = ""

    for idx, (token, word_id) in enumerate(zip(tokens, word_ids)):
        if word_id is None:
            continue

        tag = id2tag[predicted_labels[idx]]

        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append({
                    "entity": current_type,
                    "content": " ".join(current_entity)
                })
            current_type = tag[2:]
            current_entity = [token]
        elif tag.startswith("I-") and current_entity is not None and tag[2:] == current_type:
            current_entity.append(token)
        else:
            if current_entity is not None:
                entities.append({
                    "entity": current_type,
                    "content": " ".join(current_entity)
                })
            current_entity = None
            current_type = ""

    if current_entity is not None:
        entities.append({
            "entity": current_type,
            "content": " ".join(current_entity)
        })

    return entities
example_text = "双方确定了今后发展中美关系的指导方针。"
inputs = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
entities = extract_entities(example_text, predictions)
print(entities)
