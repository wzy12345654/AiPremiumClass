# -*- coding: utf-8 -*-
"""
命名实体识别 (NER) 模型训练脚本

使用 HuggingFace Transformers 和 DDP 进行分布式混合精度训练。
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate  # pip install evaluate
import seqeval   # pip install seqeval
from datasets import load_dataset


# ==============================================
# 分布式环境设置与清理
# ==============================================

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()


# ==============================================
# 数据处理函数
# ==============================================

def get_entities_tags():
    """
    定义实体标签映射关系：
    - O 表示非实体
    - B-X 表示实体X的开始
    - I-X 表示实体X的中间或结尾
    """
    entities = ['O'] + list({'movie', 'name', 'game', 'address', 'position',
                             'company', 'scene', 'book', 'organization', 'government'})
    tags = ['O']
    for entity in entities[1:]:
        tags.append(f'B-{entity.upper()}')
        tags.append(f'I-{entity.upper()}')
    return tags


def create_entity_index(tags):
    """创建实体到索引的映射"""
    return {tag: i for i, tag in enumerate(tags)}


def entity_tags_proc(item):
    """
    处理每条数据，将实体信息转换为标签序列

    Args:
        item: dataset 中的一条记录

    Returns:
        dict: 包含 `ent_tag` 标签列表
    """
    text_len = len(item['text'])
    tags = [0] * text_len  # 初始化所有标签为 'O'

    for ent in item['ents']:
        indices = ent['indices']
        label = ent['label']
        idx = create_entity_index(tags)[label]
        tags[indices[0]] = idx * 2 - 1  # B-tag
        for idx_pos in indices[1:]:
            tags[idx_pos] = idx * 2  # I-tag

    return {'ent_tag': tags}


def tokenize_and_align_labels(item):
    """
    对文本进行分词，并对齐标签

    Args:
        item: dataset 中的一条记录

    Returns:
        dict: 包含 tokenized 输入和对应标签
    """
    batch_texts = [list(text) for text in item['text']]
    tokenized_inputs = tokenizer(
        batch_texts,
        truncation=True,
        add_special_tokens=False,
        max_length=512,
        is_split_into_words=True,
        padding='max_length'
    )

    labels = []
    for i in range(len(item["ent_tag"])):
        label = item["ent_tag"][i]
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label[word_idx] != 0 else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ==============================================
# 模型评估指标计算
# ==============================================

def compute_metrics(eval_preds):
    """
    计算 NER 的评估指标（precision, recall, f1）

    Args:
        eval_preds: 预测结果 tuple(predicted_logits, labels)

    Returns:
        dict: 评估结果
    """
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tags[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


# ==============================================
# 主训练流程
# ==============================================

def train(rank, world_size):
    """在单个 GPU 上执行模型训练"""
    setup(rank, world_size)

    # 加载原始数据集
    ds = load_dataset('nlhappy/CLUE-NER')

    # 获取标签集合
    global tags
    tags = get_entities_tags()
    global entity_index
    entity_index = create_entity_index(tags)

    # 加载 Tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    # 处理数据集：添加标签列
    ds1 = ds.map(entity_tags_proc)

    # 分词并对其标签
    ds2 = ds1.map(tokenize_and_align_labels, batched=True)

    # 创建模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=len(tags),
        id2label={i: tag for i, tag in enumerate(tags)},
        label2id={tag: i for i, tag in enumerate(tags)}
    ).to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="ner_train",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        fp16=True,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        report_to='tensorboard',
        local_rank=rank,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=1,
        run_name="bert-ner-clue",
        push_to_hub=False,
        load_best_model_at_end=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds2["train"],
        eval_dataset=ds2["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    cleanup()


def main():
    """主函数，启动多进程训练"""
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

