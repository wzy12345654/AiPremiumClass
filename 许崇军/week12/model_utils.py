from transformers import AutoModelForTokenClassification
import torch.optim as optim
def build_model(id2label, label2id, model_name='bert-base-chinese'):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    return model


def get_optimizer(model, bert_lr=1e-5, cls_lr=1e-3, weight_decay=0.1):
    param_optimizer = list(model.named_parameters())
    bert_params, classifier_params = [], []
    for name, params in param_optimizer:
        if 'bert' in name:
            bert_params.append(params)
        else:
            classifier_params.append(params)

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': bert_lr},
        {'params': classifier_params, 'lr': cls_lr, 'weight_decay': weight_decay}
    ])
    return optimizer
if __name__ == '__main__':
    id2label = {0: 'O', 1: 'B-ENTITY', 2: 'I-ENTITY'}
    label2id = {v: k for k, v in id2label.items()}
    model = build_model(id2label, label2id)
    optimizer = get_optimizer(model)
    print(model)
    print("--------")
    print(optimizer)