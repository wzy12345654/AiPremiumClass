import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from data_utils import prepare_data, get_entities_tags
from model_utils import build_model, get_optimizer
from config import device

# 加载数据和标签映射
ds2,tokenizer = prepare_data()
_, tags, _, id2label, label2id = get_entities_tags()
model = build_model(id2label, label2id)
model.to(device)
# 数据加载器
train_dl = DataLoader(ds2['train'], batch_size=16, shuffle=True)
# 优化器
optimizer = get_optimizer(model, bert_lr=1e-5, cls_lr=1e-3)
# 学习率调度器
total_steps = len(train_dl) * 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
# 混合精度 scaler
scaler = torch.GradScaler()
# 开始训练
for epoch in range(5):
    model.train()
    tpbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
    for items in tpbar:
        items = {k: v.to(device) for k, v in items.items()}
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            outputs = model(**items)
        loss = outputs.loss
        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tpbar.set_description(f'Epoch:{epoch + 1} ' +
                              f'bert_lr:{scheduler.get_lr()[0]} ' +
                              f'classifier_lr:{scheduler.get_lr()[1]} ' +
                              f'Loss:{loss.item():.4f}')
