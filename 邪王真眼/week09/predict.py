import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from transformers import BertTokenizer
from models.models import get_model
from configs.config import load_train_config
from datasets.datasets import get_dataset


def generate_text():
    user_input = input("\n请输入您的问题: ")
    if user_input.lower() in ['exit', 'quit']:
        return False
    
    enc_tokens = tokenizer.encode(
        user_input,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg['encoder_max_length']
    )
    enc_input = torch.tensor([enc_tokens], device=device)
    enc_mask = torch.ones_like(enc_input, dtype=torch.bool)
    
    dec_input = torch.tensor([[tokenizer.cls_token_id]], device=device)
    
    print(f"问题: {user_input}")
    print("回答: ", end="", flush=True)
    
    for _ in range(cfg['decoder_max_length'] - 1):
        cur_len = dec_input.size(1)
        dec_mask = torch.tril(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device))
        
        with torch.no_grad():
            output = model(enc_input, dec_input, 
                         encoder_mask=enc_mask,
                         decoder_mask=dec_mask.unsqueeze(0))
            
            next_logits = output["out"][0, :, -1]
        
        next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
        
        if next_token.item() == tokenizer.sep_token_id:
            break
            
        print(tokenizer.decode(next_token.item()), end="", flush=True)
        
        dec_input = torch.cat([dec_input, next_token], dim=1)
    
    print()
    return True

if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week09.py")
    model_path = "./邪王真眼/week09/result/my_transformer.pth"
    cfg['batch_size'] = 1
    
    # dataset
    _, val_loader = get_dataset(cfg)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(cfg).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    while True:
        if not generate_text():
            break