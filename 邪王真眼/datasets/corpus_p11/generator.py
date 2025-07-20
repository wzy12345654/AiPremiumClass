import os
import json
import torch

from typing import Any, Dict, List
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class QA(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        data_root = cfg.get("data_root")
        self.max_length = cfg.get("max_length")
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.vocab = self.tokenizer.get_vocab()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        self.ans_token = "[ANS]"
        self.tokenizer.add_tokens([self.ans_token])
        self.ans_token_id = self.tokenizer.convert_tokens_to_ids(self.ans_token)
        
        cfg["vocab_size"] = self.vocab_size
        cfg["pad_token_id"] = self.pad_token_id
        
        self.data = []
        with open(os.path.join(data_root, 'sample_data.json'), 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        split_idx = int(0.7 * len(self.data))
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        self.tokenized_pairs = self._pre_tokenize_pairs()
    
    def _pre_tokenize_pairs(self) -> List[Dict[str, Any]]:
        tokenized = []
        for item in self.data:
            q_tokens = self.tokenizer.encode(
                item["title"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - 2
            )
            
            a_tokens = self.tokenizer.encode(
                item["content"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(q_tokens) - 1
            )
            
            input_ids = q_tokens + [self.cls_token_id] + a_tokens[:-1]
            input_ids = input_ids[:self.max_length]
            
            labels = [0] * len(q_tokens) + a_tokens
            labels = labels[:self.max_length]
            
            tokenized.append({
                "input_ids": input_ids,
                "labels": labels,
                "q_len": len(q_tokens),
                "real_len": len(input_ids)
            })
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.tokenized_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.tokenized_pairs[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        q_len = item["q_len"]
        real_len = item["real_len"]
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [0] * padding_length
        
        attention_mask = torch.zeros((self.max_length, self.max_length), dtype=torch.long)
        
        for i in range(q_len):
            attention_mask[i, :q_len] = 1
        
        attention_mask[q_len, :q_len] = 1
        
        for i in range(q_len + 1, real_len):
            attention_mask[i, :q_len + 1] = 1
            attention_mask[i, q_len + 1:i + 1] = 1 
        
        attention_mask[real_len:, :] = 0
        
        return {
            "x": torch.tensor(input_ids, dtype=torch.long),
            "mask": attention_mask,
            "label": torch.tensor(labels, dtype=torch.long),
            "tag": torch.tensor(labels, dtype=torch.long)
        }