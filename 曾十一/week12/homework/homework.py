import os, numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from datasets import load_dataset
import evaluate

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    setup(rank, world_size)

    dataset = load_dataset("doushabao4766/msra_ner_k_V3")
    label_list = dataset["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained(
        "google-bert/bert-base-chinese",
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label
    ).to(rank)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label[word_idx] % 2 == 1 else label[word_idx] + 1)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    dataset = dataset.remove_columns(["tokens", "ner_tags", "id", "knowledge"])
    dataset.set_format("torch")

    print(f"✅ Rank {rank} 构造 TrainingArguments")
    training_args = TrainingArguments(
        output_dir="./ner_ddp_output",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./ner_ddp_output/logs",
        logging_steps=50,
        fp16=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        lr_scheduler_type="linear",
        warmup_steps=100,
        save_total_limit=2,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        metric = evaluate.load("seqeval")
        return metric.compute(predictions=true_predictions, references=true_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    trainer.train()
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print("🔥 DDP 多进程失败：", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
