import os
import torch
import pandas as pd

from typing import Any, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class JDComments(Dataset):
    def __init__(self, split: str, cfg: Dict[str, Any]):
        self.split = split
        self.data_path = os.path.join(cfg.get("data_root"), 'jd_comment_data.xlsx')
        self.split_ratio = cfg.get("split_ratio")
        self.data_max_num = cfg.get("data_max_num", None)
        self.max_length = cfg.get("max_length")
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        
        cfg.update({
            "vocab_size": self.tokenizer.vocab_size,
            "num_classes": 5
        })
        
        self.data = self._load_data()

    def _load_data(self) -> list:
        raw = pd.read_excel(
            self.data_path,
            engine='openpyxl',
            usecols=[4, 5],
            header=1,
            names=["Star", "Comment"],
        )
        
        comment_counts = raw["Comment"].value_counts()
        repeated_comments = comment_counts[comment_counts>1].index.tolist()
        
        if repeated_comments:
            raw = raw[~raw["Comment"].isin(repeated_comments)]
        
        if self.data_max_num:
            data = raw.iloc[:self.data_max_num].reset_index(drop=True)
            print(f"数据使用量: {min(self.data_max_num, len(raw))} / {len(raw)}")
        else:
            data = raw.iloc[:].reset_index(drop=True)

        split_idx = int(len(data) * self.split_ratio)
        data = data.iloc[:split_idx] if self.split == "train" else data.iloc[split_idx:]
        
        samples = []
        for _, row in data.iterrows():
            comment = str(row["Comment"])
            star = int(row["Star"]) - 1
            
            encoding = self.tokenizer(
                comment,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True
            )
            
            samples.append({
                "x": encoding["input_ids"].squeeze(0),
                "mask": encoding["attention_mask"].squeeze(0),
                "label": torch.tensor(star, dtype=torch.long)
            })
        
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]