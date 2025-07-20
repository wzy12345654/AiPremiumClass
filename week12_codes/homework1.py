import os
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import evaluate  # pip install evaluate
import seqeval   # pip install seqeval
from datasets import load_dataset

def train(local_rank):
    # 数据集
    ds = load_dataset('ds_msra_ner')
    # entity_index
    entites = ['O'] + list({'PER', 'LOC', 'ORG'})
    tags = ['O']
    for entity in entites[1:]:
        tags.append('B-' + entity.upper())
        tags.append('I-' + entity.upper())

    entity_index = {entity:i for i, entity in enumerate(entites)}
        
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    def data_input_proc(item):
        # 输入文本先拆分为字符，再转换为模型输入的token索引
        batch_texts = [list(text) for text in item['tokens']]
        # 导入拆分为字符的文本列表时，需要设置参数is_split_into_words=True
        input_data = tokenizer(batch_texts, truncation=True, add_special_tokens=False, max_length=256, 
                               is_split_into_words=True, padding='max_length')
        input_data['labels'] = [lbl[:256] for lbl in item['ner_tags']]
        return input_data
        
    
    ds2 = ds.map(data_input_proc, batched=True)  # batch_size 1000
    
    
    id2lbl = {i:tag for i, tag in enumerate(tags)}
    lbl2id = {tag:i for i, tag in enumerate(tags)}
    
    model = AutoModelForTokenClassification.from_pretrained('bert-base-chinese', 
                                                            num_labels=len(tags),
                                                            id2label=id2lbl,
                                                            label2id=lbl2id)
    model.to(local_rank)
    
    args = TrainingArguments(
        output_dir="ner_train",  # 模型训练工作目录（tensorboard，临时模型存盘文件，日志）
        num_train_epochs = 3,    # 训练 epoch
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        per_device_train_batch_size=16,  # 训练批次
        per_device_eval_batch_size=16,
        report_to='tensorboard',  # 训练输出记录
        eval_strategy="epoch",
        local_rank=local_rank,   # 当前进程 RANK
        fp16=True,               # 使用混合精度
        lr_scheduler_type='linear',  # 动态学习率
        warmup_steps=100,        # 预热步数
        ddp_find_unused_parameters=False  # 优化DDP性能
    )
    
    def compute_metric(result):
        # result 是一个tuple (predicts, labels)
        
        # 获取评估对象
        seqeval = evaluate.load('seqeval')
        predicts,labels = result
        predicts = np.argmax(predicts, axis=2)
        
        # 准备评估数据
        predicts = [[tags[p] for p,l in zip(ps,ls) if l != -100]
                     for ps,ls in zip(predicts,labels)]
        labels = [[tags[l] for p,l in zip(ps,ls) if l != -100]
                     for ps,ls in zip(predicts,labels)]
        results = seqeval.compute(predictions=predicts, references=labels)
    
        return results
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=ds2['train'],
        eval_dataset=ds2['test'],
        data_collator=data_collator,
        compute_metrics=compute_metric
    )
    
    trainer.train()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    train(args.local_rank)

if __name__ == "__main__":
    main()