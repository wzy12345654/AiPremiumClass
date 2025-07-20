import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import seqeval
from data_utils import prepare_data, get_entities_tags
from model_utils import build_model

# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    print(f"[Rank {rank}] 进程启动，开始训练...")

    # 加载数据集和模型
    ds2, tokenizer = prepare_data()
    _, tags, _, id2label, label2id = get_entities_tags()
    model = build_model(id2label, label2id)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 构建评估函数
    def compute_metric(result):
        seqeval = evaluate.load('seqeval')
        predicts, labels = result
        predicts = np.argmax(predicts, axis=2)

        # 格式转换
        predicts = [[tags[p] for p, l in zip(ps, ls) if l != -100]
                    for ps, ls in zip(predicts, labels)]
        labels = [[tags[l] for p, l in zip(ps, ls) if l != -100]
                  for ps, ls in zip(predicts, labels)]
        results = seqeval.compute(predictions=predicts, references=labels)
        return results

    # 构建训练参数
    args = TrainingArguments(
        output_dir="ner_train",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        report_to='tensorboard',
        eval_strategy="epoch",
        local_rank=rank,
        fp16=True,
        lr_scheduler_type='linear',
        warmup_steps=100,
        ddp_find_unused_parameters=False
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model.module,
        args=args,
        train_dataset=ds2['train'],
        eval_dataset=ds2['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metric
    )
    trainer.train()
    if rank == 0:
        model.module.save_pretrained("./ner_model")  # 保存为 HuggingFace 格式
        tokenizer.save_pretrained("./ner_model")
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
