import torch
import re
from transformers import AutoModelForSequenceClassification, BertTokenizer
import pandas as pd
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model.load_state_dict(torch.load('bert_model_full.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.strip()

def predict_score(comment):
    cleaned_comment = clean_text(comment)
    if not cleaned_comment:
        return "输入内容无效，请提供有效评价文本。"

    encoding = tokenizer(
        cleaned_comment,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class + 1

# 示例调用
if __name__ == '__main__':
    user_input = input("请输入商品评价内容：")
    score = predict_score(user_input)
    print(f"预测评分为：{score} 分")
