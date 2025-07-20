import torch
import pickle
from EncoderDecoderAttenModel import Seq2Seq

def generate_couplet(model, input_text, enc_vocab, dec_vocab, device, max_length=50):
    """
    根据输入的上联生成下联
    """
    model.eval()
    with torch.no_grad():
        # 将输入文本转换为token indices
        enc_tokens = ['BOS'] + list(input_text.split()) + ['EOS']
        enc_indices = [enc_vocab[token] for token in enc_tokens]
        enc_input = torch.tensor([enc_indices]).to(device)
        
        # 编码器前向传播
        hidden_state, enc_outputs = model.encoder(enc_input)
        
        # 准备解码器输入
        dec_input = torch.tensor([[dec_vocab['BOS']]]).to(device)
        
        # 存储生成的token
        generated_tokens = []
        
        # 逐步生成
        while True:
            if len(generated_tokens) >= max_length:
                break
                
            # 解码器前向传播
            logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)
            
            # 获取概率最高的token
            next_token = torch.argmax(logits, dim=-1)
            next_token_idx = next_token.squeeze().item()
            
            # 如果生成了EOS，停止生成
            if dec_vocab.inv_vocab[next_token_idx] == 'EOS':
                break
                
            generated_tokens.append(next_token_idx)
            
            # 更新解码器输入
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)
    
    # 将token indices转换回文字
    generated_text = ''.join(dec_vocab.inv_vocab[idx] for idx in generated_tokens)
    return generated_text

if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载词表
    with open('couplet_vocab.bin', 'rb') as f:
        enc_vocab, dec_vocab = pickle.load(f)
    
    # 加载模型
    model = Seq2Seq(
        enc_emb_size=len(enc_vocab),
        dec_emb_size=len(dec_vocab),
        emb_dim=256,
        hidden_size=512,
        dropout=0.5
    )
    model.load_state_dict(torch.load('couplet_model_final.pt'))
    model.to(device)
    
    # 交互式生成
    print('欢迎使用对联生成系统！输入上联，系统将为您生成下联。输入q退出。')
    while True:
        input_text = input('请输入上联：')
        if input_text.lower() == 'q':
            break
            
        try:
            generated_text = generate_couplet(model, input_text, enc_vocab, dec_vocab, device)
            print(f'生成的下联：{generated_text}')
        except Exception as e:
            print(f'生成失败：{str(e)}')