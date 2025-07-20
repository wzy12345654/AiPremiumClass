import torch
import torch.nn as nn

from transformers import BertModel, BertConfig
from models.losses.loss_functions import get_loss_function


class BertPredictor(nn.Module):
    def __init__(self, cfg: dict):
        super(BertPredictor, self).__init__()
        vocab_size = cfg.get("vocab_size")
        num_layers = cfg.get("num_layers")
        dropout = cfg.get("dropout")

        bert_cfg = BertConfig.from_pretrained(
            "bert-base-chinese",
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            output_hidden_states=False,
            return_dict=True
        )
        self.bert = BertModel.from_pretrained("bert-base-chinese", config=bert_cfg)
        self.predict = nn.Linear(768, vocab_size)
        self.pad_token_id = cfg.get("pad_token_id")
        self.max_length = cfg.get("max_length")
        cfg['ignore_index'] = self.pad_token_id
        self.loss_fn = get_loss_function(cfg)
    
    def _process_mask(self, mask):
        if mask.dim() == 2:
            _, seq_len = mask.shape
            device = mask.device
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
            padding_mask = mask.unsqueeze(-1).bool()
            attention_mask = causal_mask.unsqueeze(0) & padding_mask
            return attention_mask.float()
        elif mask.dim() == 3:
            return mask.float()
        else:
            raise ValueError(f"Invalid mask dimension: {mask.dim()}")
        
    def forward(self, x, mask, tag=None):
        attention_mask = self._process_mask(mask)
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        logits = self.predict(last_hidden)  # (bs, seq_len, vocab_size)
        out = logits.permute(0, 2, 1).contiguous()  # (bs, vocab_size, seq_len)
        
        output = {'out': out}
        if tag is not None:
            loss = self.loss_fn(out, tag)
            output['loss'] = loss

        return output