import os
import json
import torch

from typing import Any, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class QA(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.encoder_max_len = cfg["encoder_max_length"]
        self.decoder_max_len = cfg["decoder_max_length"]
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab = self.tokenizer.vocab
        
        cfg["vocab_size"] = self.vocab_size
        cfg["pad_token_id"] = self.pad_token_id
        
        with open(os.path.join(cfg["data_root"], 'sample_data.json'), 'r', encoding='utf-8') as f:
            full_data = [json.loads(line) for line in f]
        
        split_idx = int(0.7 * len(full_data))
        self.data = full_data[:split_idx] if split == 'train' else full_data[split_idx:]
        
        self.processed_data = []
        for item in self.data:
            enc_tokens = self.tokenizer.encode(
                item["title"],
                max_length=self.encoder_max_len,
                truncation=True,
                add_special_tokens=False
            )
            enc_padded = self._pad_sequence(enc_tokens, self.encoder_max_len)
            
            ans_tokens = self.tokenizer.encode(
                item["content"],
                max_length=self.decoder_max_len-1,
                truncation=True,
                add_special_tokens=False
            )
            dec_input = [self.cls_token_id] + ans_tokens[:-1]
            labels = ans_tokens
            
            dec_padded = self._pad_sequence(dec_input, self.decoder_max_len)
            labels_padded = self._pad_sequence(labels, self.decoder_max_len)
            enc_mask = self._generate_mask(len(enc_tokens))
            dec_mask = self._generate_mask(len(dec_input))
            
            self.processed_data.append({
                "x_encoder": torch.tensor(enc_padded, dtype=torch.long),
                "x_decoder": torch.tensor(dec_padded, dtype=torch.long),
                "encoder_mask": enc_mask,
                "decoder_mask": dec_mask,
                "label": torch.tensor(labels_padded, dtype=torch.long)
            })
    
    def _pad_sequence(self, sequence: list, max_len: int) -> list:
        return sequence + [self.pad_token_id] * (max_len - len(sequence))
    
    def _generate_mask(self, valid_length: int) -> torch.Tensor:
        mask = torch.cat([
            torch.ones(valid_length, dtype=torch.bool),
            torch.zeros(self.encoder_max_len - valid_length, dtype=torch.bool)
        ])
        return mask
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]