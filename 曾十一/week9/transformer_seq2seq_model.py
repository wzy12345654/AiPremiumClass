import torch
import torch.nn as nn
import math

from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=50):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embdding = torch.zeros((maxlen, emb_size))
        pos_embdding[:, 0::2] = torch.sin(pos * den)
        pos_embdding[:, 1::2] = torch.cos(pos * den)

        pos_embdding = pos_embdding.unsqueeze(0)  
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embdding)

    def forward(self, token_embdding):
        token_len = token_embdding.size(1)
        return self.dropout(self.pos_embedding[:, :token_len, :] + token_embdding)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers, 
                 dim_forward, dropout, enc_voc_size, dec_voc_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_forward,
                                          dropout=dropout, batch_first=True)
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        self.predict = nn.Linear(d_model, dec_voc_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=enc_pad_mask, 
                                tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)

    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)

    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


