import numpy as np
import torch
import seqeval
import evaluate

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification


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


def data_input_proc(item):
    # 输入文本转换模型输入token索引
        input_data = tokenizer(item['text'], truncation=True, add_special_tokens=False, max_length=512)
        adjust_labels = []  # 所有修正后label索引列表
        # 上一步骤生成ner_tag中索引和token对齐
        for k in range(len(input_data['input_ids'])):
            # 每条记录token对应word_ids
            word_ids = input_data.word_ids(k)
            # 批次ner_tag长度和token长度对齐
            tags = item['ent_tag'][k]
            
            adjusted_label_ids = []
            i, prev_wid = -1,-1
            for wid in word_ids:
                if (wid != prev_wid):   #  word_ids [1,1,1,2,3,4,5] -> [0,1,2,3,4,5,6]
                    i += 1 # token对应检索位置+1
                    prev_wid = wid
                adjusted_label_ids.append(tags[i])
            adjust_labels.append(adjusted_label_ids)                
        # 修正后label添加到input_data
        input_data['labels'] = adjust_labels
        return input_data


def compute_metric(result):
    # result 是一个tuple (predicts, labels)
    
    # 获取评估对象
    seqeval = evaluate.load('seqeval')
    predicts, labels = result
    predicts = np.argmax(predicts, axis=2)
    
    # 准备评估数据
    predicts = [[tags[p] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    labels = [[tags[l] for p,l in zip(ps,ls) if l != -100]
                 for ps,ls in zip(predicts,labels)]
    results = seqeval.compute(predictions=predicts, references=labels)

    return results



if __name__ == "__main__":
    model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese', num_labels=7)
    # ds = load_dataset("doushabao4766/msra_ner_k_V3")
    ds = load_dataset('nlhappy/CLUE-NER')
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')

    message= "命名实体识别"
    label = torch.tensor([0,1,0,2,5,4])

    model_input = tokenizer([message], return_tensors='pt')
    result = model(**model_input)

    entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position', \
           'company', 'scene', 'book', 'organization', 'government'})
    tags = ['O']
    for entity in entites[1:]:
        tags.append('B-' + entity.upper())
        tags.append('I-' + entity.upper())

    entity_index = {entity:i for i, entity in enumerate(entites)}

    
    ds1 = ds.map(entity_tags_proc)

    for row in ds1['train']:
        print(row['text'])
        print(row['ent_tag'])
        
        break

    token_index = tokenizer.encode('2000年2月add', add_special_tokens=False)
    print(token_index)
    tokens = tokenizer.decode(token_index)
    print(tokens)

    input_data = tokenizer(['2000年2月add'], add_special_tokens=False, truncation=True)
    print(input_data)

    input_data.word_ids(0) # 返回批次中指定token对应原始文本的索引映射

    ds2 = ds1.map(data_input_proc, batched=True)

    ds2.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    args = TrainingArguments(
        output_dir="ner_train",  # 模型训练工作目录（tensorboard，临时模型存盘文件，日志）
        num_train_epochs = 3,    # 训练 epoch
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        per_device_train_batch_size=32,  # 训练批次
        per_device_eval_batch_size=32,
        report_to='tensorboard',  # 训练输出记录
        eval_strategy="epoch",
    )

    id2lbl = {i:tag for i, tag in enumerate(tags)}
    lbl2id = {tag:i for i, tag in enumerate(tags)}

    model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese', 
                                                            num_labels=21,
                                                            id2label=id2lbl,
                                                            label2id=lbl2id)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model,
        args,
        train_dataset=ds2['train'],
        eval_dataset=ds2['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metric
    )

    trainer.train()

    result = trainer.predict(ds2['validation'])

    print(ds1['validation'][10]['text'])
    print(ds2['validation'][10]['labels'])
    print(result.label_ids[10])