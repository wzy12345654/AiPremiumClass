import torch.nn as nn

from typing import Any, Dict
from transformers import AutoModelForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super(BertClassifier, self).__init__()
        num_classes = cfg.get("num_classes")
        self.bert_classifier = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=num_classes)

    def forward(self, x, mask):
        out = self.bert_classifier(
            input_ids=x,
            attention_mask=mask,
            return_dict=True
        )
        return {'out': out.logits}