# inference.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_utils import get_entities_tags

# 获取标签映射
_, tags, _, id2label, label2id = get_entities_tags()

# 加载模型和分词器
model_path = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    """
    输入文本，输出实体识别结果
    :param text: str
    :return: list of (token, entity_type)
    """
    inputs = tokenizer(
        list(text),  # 按字符切分输入
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        is_split_into_words=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    results = []

    for token, pred_id in zip(tokens, predictions):
        if token in ["[PAD]", "[CLS]", "[SEP]"]:
            continue
        label = id2label[pred_id]
        results.append((token, label))

    return results

# 示例测试
if __name__ == "__main__":
    test_text = "我爱北京天安门"
    prediction = predict(test_text)
    print(prediction)
