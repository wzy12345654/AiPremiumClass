import csv
import os
import torch
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn, optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# === Dataset 类定义 ===
class JDDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# === collate_fn 构造 ===
def build_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encodings = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        labels = torch.tensor([label for label in labels])
        encodings["labels"] = labels
        return encodings
    return collate_fn

# === 推理函数 ===
def predict(text, model_dir="./jd_model"):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1)
        return pred.item()

# === 训练函数 ===
def train_one_epoch(model, dataloader, optimizer, device, epoch, writer, prefix):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"[Train] Epoch {epoch} ({prefix})"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar(f"{prefix}/Loss", avg_loss, epoch)
    print(f"✅ Epoch {epoch} | {prefix} Train Loss: {avg_loss:.4f}")

# === 验证函数 ===
def evaluate(model, dataloader, device, epoch, writer, prefix):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            labels = inputs['labels']
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    writer.add_scalar(f"{prefix}/Accuracy", acc, epoch)
    print(f"📊 {prefix} Validation Accuracy: {acc:.4f}")
    return acc

# === 主函数入口 ===
def run_experiment(freeze_bert=False, run_name="unfrozen", args=None):
    print(f"🚀 正在启动实验：{run_name} (freeze_bert={freeze_bert})")

    # === Step 1: 加载数据 ===
    data = []
    with open(args.csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = row["评价内容(content)"].strip()
            score = row["评分（总分5分）(score)"].strip()
            if content == "此用户未填写评价内容" or not score.isdigit() or score not in {"1", "5"}:
                continue
            data.append((content, 1 if score == "5" else 0))

    texts, labels = zip(*data)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    collate_fn = build_collate_fn(tokenizer, args.max_length)
    train_dataset = JDDataset(train_texts, train_labels)
    val_dataset = JDDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    if freeze_bert:
        for param in model.bert.parameters():
            param.requires_grad = False
        print("🚫 冻结 BERT 编码层参数。")

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, writer, prefix=run_name)
        evaluate(model, val_loader, device, epoch, writer, prefix=run_name)

    model_path = os.path.join(args.save_dir, run_name)
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    writer.close()
    print(f"✅ 实验 {run_name} 完成，模型保存在 {model_path}")

# === 实验对比入口 ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/mnt/data_1/zfy/self/homework/10/jd_comment_data.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="./jd_model")
    parser.add_argument("--log_dir", type=str, default="./runs")
    args = parser.parse_args()

    #run_experiment(freeze_bert=True, run_name="frozen", args=args)
    run_experiment(freeze_bert=False, run_name="unfrozen", args=args)

if __name__ == "__main__":
    main()
