import os
import re
import torch

from typing import Any, Dict, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NextWordPrediction(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        data_root = cfg.get("data_root")
        self.max_length = cfg.get("max_length")

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.vocab = self.tokenizer.get_vocab()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.tokenizer.vocab_size
        cfg["vocab_size"] = self.vocab_size
        cfg["pad_token_id"] = self.pad_token_id

        with open(os.path.join(data_root, 'corpus.txt'), encoding="gb18030") as f:
            text = f.read().replace("\n", "").replace(" ", "")

        sentences = self._split_sentences(text)
        self.sentences = [s for s in sentences if len(s) > 0]

        split_idx = int(0.7 * len(self.sentences))
        if split == 'train':
            self.sentences = self.sentences[:split_idx]
        else:
            self.sentences = self.sentences[split_idx:]

        self.tokenized_sentences = self._pre_tokenize_sentences()

    def _split_sentences(self, text: str) -> List[str]:
        pattern = re.compile(r'([^。！？]*[。！？])')
        sentences = pattern.findall(text)
        return sentences

    def _pre_tokenize_sentences(self) -> List[List[int]]:
        tokenized = []
        for sent in self.sentences:
            tokens = self.tokenizer.encode(
                sent,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 2
            )
            tokenized.append(tokens)
        return tokenized

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.tokenized_sentences[idx]
        
        input_ids = [self.cls_token_id] + tokens[:-1]
        labels = tokens[:]

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [self.pad_token_id] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        attention_mask = [1 if token != self.pad_token_id else 0 for token in input_ids]

        return {
            "x": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(labels, dtype=torch.long),
            "tag": torch.tensor(labels, dtype=torch.long)
        }
