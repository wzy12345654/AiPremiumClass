import torch
MAX_LENGTH = 512
BATCH_SIZE = 16
BERT_LR = 1e-5
CLS_LR = 1e-3
WEIGHT_DECAY = 0.1
EPOCHS = 5
WARMUP_STEPS = 100
MODEL_NAME = 'bert-base-chinese'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
