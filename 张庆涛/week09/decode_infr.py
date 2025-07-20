import torch
from transformer_model import Seq2SeqTransformer
from train import build_vocab, generate_square_subsequent_mask  # 添加这行

# 贪婪解码 （所有获取结果中，取概率最大的值）
def greedy_decode (model,enc_input, enc_vocab, dec_vocab, inv_dec_vocab, device, max_len=20):
  model.eval()
  enc_input = torch.tensor([[enc_vocab.get(w, 0) for w in enc_input]], dtype=torch.long).to(device)

  enc_pad_mask = (enc_input == 0) # (1,seq_len)
  memory = model.encode(enc_input) # (1,seq_len,emb_size)
  ys = torch.tensor([[dec_vocab['<s>']]], dtype=torch.long).to(device)
  for i in range(max_len):
    dec_pad_mask = (ys == 0) # (1,seq_len)
    tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
    out = model.decode(ys, memory, tgt_mask) # (1,seq_len,emb_size)
    out = model.predict(out)[:, -1, :] # 取最后一个时间步骤
    prob = out.softmax(-1) # (1,seq_len,dec_voc_size)
    next_token = prob.argmax(-1).item() # (1,seq_len)
    ys = torch.cat([ys, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    if next_token == dec_vocab['</s>']:
      break
  # 去除掉<s>和</s>
  result = [inv_dec_vocab[idx] for idx in ys[0].cpu().numpy()]
  if result[0] == '<s>':
    result = result[1:]
  if '</s>' in result:
    result = result[:result.index('</s>')]
  return ''.join(result)
    
if __name__ == '__main__':
    # 模型数据
    # 一批语料： encoder：decoder
    # <s></s><pad>
    corpus= "人生得意须尽欢，莫使金樽空对月"
    chs = list(corpus)
    enc_tokens, dec_tokens = [],[]
    for i in range(1,len(chs)):
      enc = chs[:i]
      dec = ['<s>'] + chs[i:] + ['</s>']
      enc_tokens.append(enc)
      dec_tokens.append(dec)
    # 构建词典
    enc_vocab = build_vocab(enc_tokens)
    dec_vocab = build_vocab(dec_tokens)
    inv_dec_vocab = { v:k for k,v in dec_vocab.items()}    

   
    # 模型参数
    d_model = 32
    nhead = 4
    num_enc_layers = 2
    num_dec_layers = 2
    dim_forward = 64
    dropout = 0.1
    enc_voc_size = len(enc_vocab)
    dec_voc_size = len(dec_vocab)
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2SeqTransformer(d_model, nhead,num_enc_layers,num_dec_layers,dim_forward,dropout,enc_voc_size,dec_voc_size).to(device)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('transformer_model.pth'))
    model.eval()
    
    # 推理演示
    test_enc = list("人生得意须尽欢，莫使")
    output = greedy_decode(model, test_enc, enc_vocab, dec_vocab, inv_dec_vocab, device)
    print(f"输入: {''.join(test_enc)}")
    print(f"输出: {output}")
    
    