import torch
from data_loader import CoupletDataset
from model import Encoder, Decoder, Seq2Seq
from config import config  

def load_model_and_vocab(model_path):
    """加载训练好的模型和词汇表"""
    # 加载训练集获取词汇表
    train_dataset = CoupletDataset('./couplet/train/in.txt', './couplet/train/out.txt')
    vocab = train_dataset.vocab
    rev_vocab = train_dataset.rev_vocab
    
    # 初始化模型结构
    encoder = Encoder(len(vocab)).to(config['device'])  
    decoder = Decoder(len(vocab)).to(config['device'])
    model = Seq2Seq(encoder, decoder).to(config['device'])
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=config['device']))  
    model.eval()
    
    return model, vocab, rev_vocab

def preprocess_input(input_str, vocab):
    """预处理输入的上联"""
    # 分词并添加特殊标记
    tokens = ['<sos>'] + input_str.split() + ['<eos>']
    # 转换为索引
    ids = [vocab.get(token, 3) for token in tokens]  # 未知词用<unk>
    return torch.LongTensor(ids).unsqueeze(1).to(config['device'])  # 添加batch维度

def generate_couplet(model, src_tensor, rev_vocab, max_len=50):
    """生成下联"""
    with torch.no_grad():
        # 编码器前向传播
        src_len = torch.tensor([src_tensor.size(0)]).to(config['device'])  
        encoder_outputs, (hidden, cell) = model.encoder(src_tensor, src_len)  # 补全编码过程
        
        # 解码器初始化
        trg_ids = [1]  # <sos>的索引
        input_tensor = torch.LongTensor([trg_ids[-1]]).to(config['device'])  
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_tensor, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_ids.append(pred_token)
            if pred_token == 2:  # 遇到<eos>停止
                break
            input_tensor = torch.LongTensor([pred_token]).to(config['device'])  
    
    # 转换索引到文字
    couplet = [rev_vocab.get(i, '<unk>') for i in trg_ids[1:-1]]  # 去除<sos>和<eos>
    return ' '.join(couplet)

if __name__ == "__main__":
    model, vocab, rev_vocab = load_model_and_vocab("best_model.pth")
    
    while True:
        input_str = input("请输入上联（输入q退出）：").strip()
        if input_str.lower() == 'q':
            break
        
        # 预处理
        src_tensor = preprocess_input(input_str, vocab)
        
        # 生成下联
        output_str = generate_couplet(model, src_tensor, rev_vocab)
        
        print(f"生成的下联：{output_str}")
        print("-"*60)