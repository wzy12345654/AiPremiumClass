import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from models.models import get_model
from transformers import BertTokenizer
from configs.config import load_train_config


def val():
    user_input = input("\n请输入评论: ")
    if user_input.lower() in ['exit', 'quit']:
        return False
    
    if user_input == "":
        return True
    
    input_ids = tokenizer.encode(
        user_input,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg['max_length']
    )
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    cur_len = input_tensor.shape[1]
    mask = torch.ones((1, cur_len), device=device).long()
    
    with torch.no_grad():
        output = model(input_tensor, mask)
        logits = output["out"]
        star = torch.argmax(logits, dim=1).item() + 1
    
    print(f"评论星级  : {star}")
    return True


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/week10.py")
    model_path = "./邪王真眼/week10/result/bert_classifier.pth"
    cfg['batch_size'] = 1
    cfg['num_classes'] = 5
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    while True:
        if not val():
            break
