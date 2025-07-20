import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import random

from models.models import get_model
from transformers import BertTokenizer
from datasets.datasets import get_dataset
from configs.config import load_train_config


def generate_text():
    vocab = [i for i in range(tokenizer.vocab_size)
            if i not in tokenizer.all_special_ids]
    start_token = random.choice(vocab)
    
    input_ids = torch.tensor([[start_token]]).to(device)
    mask = torch.ones_like(input_ids)
    
    print(f"起始字: {tokenizer.decode(start_token)}", end="", flush=True)
    
    with torch.no_grad():
        output = model(input_ids, mask)
        logits = output["out"]  # [1, vocab_size, seq_len]
    
    pred_tokens = torch.argmax(logits, dim=1).squeeze(0)  # [seq_len]
    generated_text = tokenizer.decode(pred_tokens.tolist())
    
    print(generated_text)


if __name__ == "__main__":
    cfg = load_train_config("./邪王真眼/configs/previous_class10.py")
    model_path = "./邪王真眼/previous_class10/result/bert_predictor.pth"
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
    
    while True:
        generate_text() # 逗号正常结果