import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, BertTokenizer
import re
from torch.utils.tensorboard import SummaryWriter


data = pd.read_excel('jd_comment_data.xlsx')
data = data[data['评价内容(content)'] != '此用户未填写评价内容']
scores = data['评分（总分5分）(score)']
contents = data['评价内容(content)']
stopwords = set(["的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "很", "呢", "啊", "哦"])
def clean_text(text):
    if pd.isna(text):  # 检查是否为 NaN
        return ''
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    text = ''.join([char for char in text if char not in stopwords])
    return text.strip()
data['content_cleaned'] = data['评价内容(content)'].apply(clean_text)
class JDCommentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        score = self.data.iloc[idx]['评分（总分5分）(score)'].astype(int) - 1  # 转换为0-4的整数
        content = self.data.iloc[idx]['content_cleaned']
        return {
            'score': score,
            'content': content
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
def build_collate(tokenizer):
    def collate_fn(batch):
        texts = [item['content'] for item in batch]
        scores = [item['score'] for item in batch]
        model_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                                 max_length=512)
        labels = torch.tensor(scores)
        return model_inputs, labels

    return collate_fn


dataset = JDCommentDataset(data)
dl = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=build_collate(tokenizer))
model = AutoModelForSequenceClassification.from_pretrained('/bert-base-chinese', num_labels=5)
# 仅冻结 BERT 主干参数，保留分类头可训练
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
writer = SummaryWriter()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    for epoch in range(5):
        progress_bar = tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
        total_loss = 0
        for batch_idx, batch in enumerate(progress_bar):
            model.zero_grad()
            model_inputs, labels = batch
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            outputs = model(**model_inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(dl) + batch_idx)
            progress_bar.set_postfix(loss=loss.item())
        epoch_loss = total_loss / len(dl)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
    torch.save(model.state_dict(), 'bert_model_full.pth')
    writer.close()
