import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
    GradScaler
)
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

# 定义数据处理函数
def entity_tags_proc(item):

def data_input_proc(item):


# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 主训练函数
def main():
    # 多GPU配置检测
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("至少需要2块GPU进行分布式训练")
        return

    # 初始化分布式进程组
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

# 单个GPU训练工作进程
def main_worker(gpu, world_size):
    setup(gpu, world_size)
    
    try:
        # 加载数据集
        ds = load_dataset('nlhappy/CLUE-NER')
        
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,  # 启用混合精度
            ddp_find_unused_parameters=False,
            report_to="tensorboard"
        )

        # 创建数据整理器
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds2["train"],
            eval_dataset=ds2["validation"],
            compute_metrics=compute_metric
        )

        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model("distilled_model")
        
    finally:
        cleanup()

# 分布式推理函数
def distributed_inference(model_path, rank, world_size):
    setup(rank, world_size)
    try:
        # 加载模型
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(rank)
        model.eval()
        
        # 构建测试数据
        test_text = ["示例测试句子"]
        test_item = tokenizer(test_text, truncation=True, is_split_into_words=True, return_tensors="pt").to(rank)
        
        # 执行推理
        with torch.no_grad():
            outputs = model(**test_item)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
        # 处理输出
        print(f"GPU {rank} 预测结果：{predictions.cpu().numpy()}")
        
    finally:
        cleanup()

# 运行分布式推理
if __name__ == "__main__":
    # 先启动训练
    main()
    
    # 再启动推理
    world_size = torch.cuda.device_count()
    mp.spawn(distributed_inference, args=("distilled_model", world_size), nprocs=world_size, join=True)

from transformers import SchedulerType

class CustomScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self):
        if self.current_step < self.warmup_steps:
            lr = (self.total_steps - self.warmup_steps) / (self.total_steps * self.warmup_steps) * self.current_step
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1

# 使用示例
scheduler = CustomScheduler(optimizer, warmup_steps=100, total_steps=1000)
scaler = GradScaler()

for inputs, labels in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

model = DDP(model, device_ids=[rank])
optimizer = optim.SGD(model.parameters(), lr=0.01)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

pip install transformers datasets evaluate seqeval torch torchvision
  dataset_name="nlhappy/CLUE-NER"
dataset = load_dataset(dataset_name)
dataset.save_to_disk("./.DS_Store")  # 提前缓存数据集

# 启动训练
python train.py

# 启动推理
python inference.py
pip install accelerate
accelerate config
from accelerate import Accelerator

accelerator = Accelerator(fp16=True, mixed_precision="bf16")
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-chinese")
config.tie_word_embeddings = True  # 减少参数量
model = AutoModelForTokenClassification(config)
