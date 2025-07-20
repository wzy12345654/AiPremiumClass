from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import evaluate  # pip install evaluate
import seqeval   # pip install seqeval
from datasets import load_dataset

model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese', num_labels=7)
message= "命名实体识别"
label = torch.tensor([0,1,0,2,5,4])

model_input = tokenizer([message], return_tensors='pt')
result = model(**model_input)

print(result.loss)
print(result.logits)
ds = load_dataset('nlhappy/CLUE-NER')
# entity_index
entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position', \
           'company', 'scene', 'book', 'organization', 'government'})
tags = ['O']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' + entity.upper())

entity_index = {entity:i for i, entity in enumerate(entites)}

# entity_index
entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position', \
           'company', 'scene', 'book', 'organization', 'government'})
tags = ['O']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' + entity.upper())

entity_index = {entity:i for i, entity in enumerate(entites)}


def entity_tags_proc(item):
    # item即是dataset中记录
    text_len = len(item['text'])  # 根据文本长度生成tags列表
    tags = [0] * text_len    # 初始值为‘O’
    # 遍历实体列表，所有实体类别标记填入tags
    entites = item['ents']
    for ent in entites:
        indices = ent['indices']  # 实体索引
        label = ent['label']   # 实体名
        tags[indices[0]] = entity_index[label] * 2 - 1
        for idx in indices[1:]:
            tags[idx] = entity_index[label] * 2
    return {'ent_tag': tags}

# 使用自定义回调函数处理数据集记录
ds1 = ds.map(entity_tags_proc)

token_index = tokenizer.encode('2000年2月add', add_special_tokens=False)
print(token_index)
tokens = tokenizer.decode(token_index)
print(tokens)

input_data = tokenizer([list('2000年2月add')], add_special_tokens=False, truncation=True,
                       is_split_into_words=True)
print(input_data)

tokens = tokenizer.decode(token_index)
print(tokens) # 返回token对应逐个字符

entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position', \
                        'company', 'scene', 'book', 'organization', 'government'})
tags = ['O']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' + entity.upper())

entity_index = {entity: i for i, entity in enumerate(entites)}


def entity_tags_proc(item):
    # item即是dataset中记录
    text_len = len(item['text'])  # 根据文本长度生成tags列表
    tags = [0] * text_len  # 初始值为‘O’
    # 遍历实体列表，所有实体类别标记填入tags
    entites = item['ents']
    for ent in entites:
        indices = ent['indices']  # 实体索引
        label = ent['label']  # 实体名
        tags[indices[0]] = entity_index[label] * 2 - 1
        for idx in indices[1:]:
            tags[idx] = entity_index[label] * 2
    return {'ent_tag': tags}


# 使用自定义回调函数处理数据集记录
ds1 = ds.map(entity_tags_proc)


def data_input_proc(item):
    # 输入文本先拆分为字符，再转换为模型输入的token索引
    batch_texts = [list(text) for text in item['text']]
    # 导入拆分为字符的文本列表时，需要设置参数is_split_into_words=True
    input_data = tokenizer(batch_texts, truncation=True, add_special_tokens=False, max_length=512,
                           is_split_into_words=True, padding='max_length')
    input_data['labels'] = [tag + [0] * (512 - len(tag)) for tag in item['ent_tag']]
    return input_data


ds2 = ds1.map(data_input_proc, batched=True)  # batch_size 1000

# 记录转换为pytorch
ds2.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

train_dl = DataLoader(ds2['train'], shuffle=True, batch_size=16)


# 模型创建
id2lbl = {i:tag for i, tag in enumerate(tags)}
lbl2id = {tag:i for i, tag in enumerate(tags)}

model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese',
                                                        num_labels=21,
                                                        id2label=id2lbl,
                                                        label2id=lbl2id)
model.to('cuda')

# 模型参数分组
param_optimizer = list(model.named_parameters())
bert_params, classifier_params = [],[]

for name,params in param_optimizer:
    if 'bert' in name:
        bert_params.append(params)
    else:
        classifier_params.append(params)

param_groups = [
    {'params':bert_params, 'lr':1e-5},
    {'params':classifier_params, 'weight_decay':0.1, 'lr':1e-3}
]

# optimizer
optimizer = optim.AdamW(param_groups) # 优化器

# 学习率调度器
train_steps = len(train_dl) * 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=100,
                                            num_training_steps=train_steps)

from tqdm import tqdm

DEVICE = 'cuda'

for epoch in range(5):
    model.train()
    tpbar = tqdm(train_dl)
    for items in tpbar:
        items = {k: v.to(DEVICE) for k, v in items.items()}
        optimizer.zero_grad()
        outputs = model(**items)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        tpbar.set_description(f'Epoch:{epoch + 1} ' +
                              f'bert_lr:{scheduler.get_lr()[0]} ' +
                              f'classifier_lr:{scheduler.get_lr()[1]} ' +
                              f'Loss:{loss.item():.4f}')

# dataLoader
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

train_dl = DataLoader(ds2['train'], shuffle=True, batch_size=16)


# 模型创建
id2lbl = {i:tag for i, tag in enumerate(tags)}
lbl2id = {tag:i for i, tag in enumerate(tags)}

model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese',
                                                        num_labels=21,
                                                        id2label=id2lbl,
                                                        label2id=lbl2id)
model.to('cuda')

# 模型参数分组
param_optimizer = list(model.named_parameters())
bert_params, classifier_params = [],[]

for name,params in param_optimizer:
    if 'bert' in name:
        bert_params.append(params)
    else:
        classifier_params.append(params)

param_groups = [
    {'params':bert_params, 'lr':1e-5},
    {'params':classifier_params, 'weight_decay':0.1, 'lr':1e-3}
]

# optimizer
optimizer = optim.AdamW(param_groups) # 优化器

# 学习率调度器
train_steps = len(train_dl) * 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=100,
                                            num_training_steps=train_steps)

from tqdm import tqdm
import torch

DEVICE = 'cuda'

# 梯度计算缩放器
scaler = torch.GradScaler()

for epoch in range(5):
    model.train()
    tpbar = tqdm(train_dl)
    for items in tpbar:
        items = {k: v.to(DEVICE) for k, v in items.items()}
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda'):
            outputs = model(**items)
        loss = outputs.loss

        # 缩放loss后，调用backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tpbar.set_description(f'Epoch:{epoch + 1} ' +
                              f'bert_lr:{scheduler.get_lr()[0]} ' +
                              f'classifier_lr:{scheduler.get_lr()[1]} ' +
                              f'Loss:{loss.item():.4f}')

%%writefile
ddp_simple.py

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# 清理分布式环境
def cleanup():
    dist.destroy_process_group()


# 定义训练循环
def train(rank, world_size):
    setup(rank, world_size)

    # 定义模型并将其移动到对应的 GPU 设备端
    model = models.resnet50().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 损失函数及优化器
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 定义数据集Dataset的转换和图像增强
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 分布式训练采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 在训练开始时创建一次
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(10):
        ddp_model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()

            #             loss.backward()
            #             optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()