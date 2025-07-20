# predict_demo.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict(text, model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1)
        return pred.item()

if __name__ == "__main__":
    model_path = "/mnt/data_1/zfy/self/homework/10/jd_model/unfrozen"
    text = "感觉东西很不错，偶尔有点小毛病，但是我还是很喜欢这家店的服务和质量。"
    result = predict(text, model_dir=model_path)
    print(f"推理文本：{text}")
    print(f"预测结果：{'好评' if result == 1 else '差评'}")
