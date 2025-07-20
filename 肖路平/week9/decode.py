import torch
from transformer_model import Seq2SeqTransformer
from train import build_vocab,generate_square_subsequent_mask

def greedy_decode(model, enc_input,enc_vocab,dec_vocab,inv_dec_vocab,device,max_len=20):
    model.eval()
    enc_input =torch.tensor([[enc_vocab.get(t,0) for t in enc_input]],dtype=torch.long).to(device)
    enc_pad_mask = (enc_input==0)
    memory = model.encode(enc_input)
    ys= torch.tensor([[dec_vocab.get('<s>',0)]],dtype=torch.long).to(device)
    
    for i in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        dec_pad_mask = (ys==0)
        out = model.decode(ys,memory,tgt_mask)
        out = model.predict(out)[:,-1,:]
        prob = out.softmax(dim=-1)
        next_token=prob.argmax(dim=-1).item()
        ys =torch.cat([ys,torch.tensor([[next_token]],dtype=torch.long).to(device)],dim=1)
        if next_token==dec_vocab.get('<s>'):
            break

        result =[inv_dec_vocab[idx] for idx in ys[0].cpu().numpy()]
        if result[0]=='</s>':
            result = result[1:]
        if '</s>' in result:
            result = result[:result.index('</s>')]

        return ''.join(result)
    
    if  __name__ == '__main__':
        corpus="天生我材必用用，千金散尽还复来"
        chs = list(corpus)

        enc_tokens,dec_tokens = [],[]

        