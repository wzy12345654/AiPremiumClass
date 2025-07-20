import torch
from transformer import Seq2SeqTransformer
def predict(model, input_text, enc_vocab, dec_vocab, device, max_length=50):
    # 将输入文本转换为token
    input_tokens = list(input_text)
    
    # 将token转换为索引
    enc_input = torch.tensor([[enc_vocab[token] for token in input_tokens]], device=device)
    
    # 初始化decoder输入（只包含开始标记）
    dec_input = torch.tensor([[dec_vocab['<s>']]], device=device)
    
    # 创建mask
    enc_pad_mask = torch.zeros((1, len(input_tokens)), dtype=torch.bool, device=device)
    
    # 开始生成
    for _ in range(max_length):
        # 创建decoder的mask（上三角矩阵）
        seq_len = dec_input.size(1)
        dec_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
        
        # 创建decoder的padding mask
        dec_pad_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        
        # 获取模型输出
        with torch.no_grad():
            output = model(enc_input, dec_input, dec_mask, enc_pad_mask, dec_pad_mask)
        
        # 获取最后一个时间步的预测
        next_token_logits = output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 如果预测到结束标记，就停止生成
        if next_token.item() == dec_vocab['</s>']:
            break
            
        # 将预测的token添加到decoder输入中
        dec_input = torch.cat([dec_input, next_token], dim=1)
    
    # 将预测的索引转换回token
    idx_to_token = {v: k for k, v in dec_vocab.items()}
    predicted_tokens = [idx_to_token[idx.item()] for idx in dec_input[0]]
    
    # 移除开始标记并返回预测结果
    return ''.join(predicted_tokens[1:])

# 主函数
def main():
    # 加载保存的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_data = torch.load('vocab_data.pth')
    enc_vocab = vocab_data['enc_vocab']
    dec_vocab = vocab_data['dec_vocab']
    # 创建模型实例（使用与训练时相同的参数）
    model = Seq2SeqTransformer(
        d_model=512,
        n_head=8,
        num_enc_layers=6,
        num_dec_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        enc_voc_size=len(enc_vocab),  # 确保使用训练时的词汇表大小
        dec_voc_size=len(dec_vocab)   # 确保使用训练时的词汇表大小
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load('transformer_model.pth'))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 测试预测
    test_input = "人生得意须尽欢"
    predicted_output = predict(model, test_input, enc_vocab, dec_vocab, device)
    print(f"输入: {test_input}")
    print(f"预测输出: {predicted_output}")

if __name__ == '__main__':
    main()
