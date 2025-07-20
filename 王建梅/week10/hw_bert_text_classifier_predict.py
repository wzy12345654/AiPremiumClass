"""
使用训练好的模型 bert_comment_classifier 进行推理分类
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_predict(text, model_name):
    # 加载预训练模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 二分类任务
    model.eval()  # 设置为评估模式

    # 对测试文本进行分类
    predictions = []
    for text in test_texts:
        # 分词和编码
        encoding = tokenizer(text, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # 模型预测
        with torch.no_grad():  # 禁用梯度计算
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()  # 获取预测类别
            predictions.append(prediction)

    # 打印预测结果
    for text, prediction in zip(test_texts, predictions):
        print(f"文本: {text}")
        print(f"预测类别: {'好评' if prediction == 1 else '差评'}")

if __name__ == '__main__':
    # 配置参数
    model_frozen = 'data/bert_comment_classifier_frozen'
    model_unfrozen  = 'data/bert_comment_classifier_unfrozen'

    # 自定义测试文本
    test_texts = [
        "这个手机质量非常好，屏幕清晰，拍照效果也非常好。",
        "这个手机的电池寿命非常短，每次充电都需要几个小时。", 
    ]

    # 进行预测
    print("frozen model predict:")
    model_predict(test_texts, model_frozen)
    print("unfrozen model predict:")
    model_predict(test_texts, model_unfrozen)


    


