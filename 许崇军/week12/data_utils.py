from datasets import load_dataset
from transformers import AutoTokenizer

def get_entities_tags(): # 获取实体标签
    entites = ['O'] + list({'movie', 'name', 'game', 'address', 'position',
                            'company', 'scene', 'book', 'organization', 'government'})
    tags = ['O']
    for entity in entites[1:]:
        tags.append(f'B-{entity.upper()}')
        tags.append(f'I-{entity.upper()}')
    entity_index = {e: i for i, e in enumerate(entites)}
    id2label = {i: t for i, t in enumerate(tags)}
    label2id = {t: i for i, t in enumerate(tags)}
    return entites, tags, entity_index, id2label, label2id
def entity_tags_proc(item, entity_index):  # 处理标签
    text_len = len(item['text'])
    tags = [0] * text_len
    ents = item.get('ents', [])
    for ent in ents:
        indices = ent['indices']
        label = ent['label']
        if label in entity_index:
            tags[indices[0]] = entity_index[label] * 2 - 1
            for idx in indices[1:]:
                tags[idx] = entity_index[label] * 2
    return {'ent_tag': tags}


def data_input_proc(item, tokenizer, max_length=512):
    batch_texts = [list(text) for text in item['text']]
    input_data = tokenizer(batch_texts, truncation=True, add_special_tokens=False,
                           max_length=max_length, is_split_into_words=True, padding='max_length')
    input_data['labels'] = [tag + [0] * (max_length - len(tag)) for tag in item['ent_tag']]
    return input_data


def prepare_data(tokenizer_name='bert-base-chinese', max_length=512):
    _, _, entity_index, _, _ = get_entities_tags()
    ds = load_dataset('nlhappy/CLUE-NER')
    ds1 = ds.map(lambda x: entity_tags_proc(x, entity_index))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds2 = ds1.map(lambda x: data_input_proc(x, tokenizer, max_length), batched=True)
    ds2.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    return ds2, tokenizer
if __name__ == '__main__':
    ds2, tokenizer = prepare_data()
    print(ds2['train'][0])
    # for item in ds2['train']:
    #     print(len(item['input_ids']), len(item['token_type_ids']), len(item['attention_mask']), len(item['labels']))
    #     break
