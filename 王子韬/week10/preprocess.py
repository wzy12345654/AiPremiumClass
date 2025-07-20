import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

def get_data():

    comment_data = pd.read_excel('jd_comment_data.xlsx')
    comment_data = comment_data[['评分（总分5分）(score)', '评价内容(content)']]
    # 删除未评价的数据
    comment_data = comment_data[
        ~ comment_data['评价内容(content)'].isin(['您没有填写内容，默认好评', '此用户未填写评价内容'])]

    # 删除不包含中文的数据
    pattern = re.compile(r'[\u4e00-\u9fff]')
    comment_data = comment_data[
        comment_data['评价内容(content)'].apply(lambda x: bool(pattern.search(str(x))))
    ]
    print(comment_data)
    return comment_data


class CommentDataset(Dataset):
    def __init__(self, data, tokenizer, label_col='评分（总分5分）(score)', text_col='评价内容(content)'):
        self.data = data
        self.tokenizer = tokenizer
        self.label_col = label_col
        self.text_col = text_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][self.text_col]
        label = int(self.data.iloc[idx][self.label_col]) - 1
        return text, label


def build_collate(tokenizer):
    def collate_fn(batch):
        sentents, labels = zip(*batch)
        model_inputs = tokenizer(list(sentents), return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(labels)
        model_inputs['labels'] = labels
        return model_inputs
    return collate_fn


def create_dataloader(comment_data, tokenizer, batch_size=8):
    dataset = CommentDataset(comment_data, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=build_collate(tokenizer))


if __name__ == '__main__':
    comment_data = get_data()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    dataloader = create_dataloader(comment_data, tokenizer)

    # 5. 创建模型，损失函数、优化器
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = CrossEntropyLoss()

    # 6. 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            loss = loss_fn(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

    # 8. 模型保存
    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')
