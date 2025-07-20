import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from models.models import get_model
from transformers import BertTokenizer
from datasets.datasets import get_dataset
from configs.config import load_train_config


def generate_text():
    user_input = input("\n请输入您的问题: ")
    if user_input.lower() in ['exit', 'quit']:
        return False
    
    q_tokens = tokenizer.encode(
        user_input,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg['max_length'] - 2
    )
    input_ids = q_tokens + [tokenizer.cls_token_id]
    
    input_tensor = torch.tensor([input_ids]).to(device)
    
    print(f"问题: {user_input}")
    print("回答: ", end="", flush=True)
    
    for _ in range(1000 - len(input_ids)):
        cur_len = input_tensor.shape[1]
        mask = torch.ones((1, cur_len, cur_len), device=device).long()
        
        with torch.no_grad():
            output = model(input_tensor, mask)
            next_logits = output["out"][0, :, -1]
        
        next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
        
        if next_token.item() == tokenizer.sep_token_id:
            break
            
        print(tokenizer.decode(next_token.item()), end="", flush=True)
        
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
    
    print()
    return True


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class11.py")
    model_path = "./邪王真眼/previous_class11/result/bert_predictor.pth"
    cfg['batch_size'] = 1

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # dataset
    _, val_loader = get_dataset(cfg)

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    while True:
        if not generate_text():
            break