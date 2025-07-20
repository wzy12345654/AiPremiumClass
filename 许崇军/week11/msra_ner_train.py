
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
import torch
import evaluate
import numpy as np
import seqeval

data_dir = 'msra_ner_k_V3'
data_files = {
    "train": f"{data_dir}/train-00000-of-00001-42717a92413393f9.parquet",
    "test": f"{data_dir}/test-00000-of-00001-8899cab5fdab45bc.parquet"
}

ds = load_dataset('parquet', data_files=data_files)
labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
tag2id = {tag: i for i, tag in enumerate(labels)}
id2tag = {i: tag for i, tag in enumerate(labels)}
model_name = "./bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2tag, label2id=tag2id)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = ds.map(tokenize_and_align_labels, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
seqeval_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
args = TrainingArguments(
    output_dir="ner_results",
    eval_strategy="epoch",
    logging_dir="logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="tensorboard",
    run_name="ner-training-msra" ,
    save_safetensors= False
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    processing_class= tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
example_text = "双方确定了今后发展中美关系的指导方针。"
inputs = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")
np.save("./ner_model/predictions.npy", predictions.numpy())
