import json
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from EncoderDecoderAttention import Sqe2seq
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_file(x,y):
    def get_data(data):
        encoder,decoder,label =[],[],[]
        for e,d in data:
            e_id = [x.get(temp,x['UNK']) for temp in e]
            d_id = [y.get(temp,y['UNK']) for temp in d]

            encoder.append(torch.tensor(e_id))
            decoder.append(torch.tensor(d_id[:-1]))
            label.append(torch.tensor(d_id[1:]))

        pad_e = pad_sequence(encoder, batch_first=True, padding_value=x['PAD'])
        pad_d = pad_sequence(decoder, batch_first=True, padding_value=y['PAD'])
        pad_t = pad_sequence(label, batch_first=True, padding_value=y['PAD'])
        return pad_e,pad_d,pad_t
    return get_data



if __name__ == '__main__':

    devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('../couplet/enc_data_train.json','r') as f:
        enc_train = json.load(f)

    with open('../couplet/dec_data_train.json','r') as f:
        dec_train = json.load(f)

    with open('../couplet/enc_data_test.json','r') as f:
        enc_test = json.load(f)

    with open('../couplet/dec_data_test.json','r') as f:
        dec_test = json.load(f)

    with open('../couplet/vocab_train.pkl', 'rb') as f:
        enc_train_vocab,dec_train_vocab = pickle.load(f)

    with open('../couplet/vocab_test.pkl', 'rb') as f:
        enc_test_vocab,dec_test_vocab = pickle.load(f)


    dataset_train = list(zip(enc_train,dec_train))
    dataset_test = list(zip(enc_test,dec_test))

    train_dataloader = DataLoader(dataset_train,batch_size=256,shuffle=True,collate_fn=get_file(enc_test_vocab,dec_test_vocab))
    test_dataloader = DataLoader(dataset_test,batch_size=256,shuffle=False,collate_fn=get_file(enc_test_vocab,dec_test_vocab))

    model = Sqe2seq(
        len(enc_train_vocab),
        len(dec_train_vocab),
        emb_size=100,hidden_size=120)
    model.to(devices)
    criterion = nn.CrossEntropyLoss().to(devices)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    writer = SummaryWriter(log_dir='runs/my_experiment')
    for epoch in range(20):
        train_global,test_global = 0,0
        model.train()
        tpbar = tqdm(train_dataloader)
        for encoder,decoder,target in tpbar:
            train_encoder = encoder.to(devices)
            train_decoder = decoder.to(devices)
            train_target = target.to(devices)
            output,_ = model(train_encoder,train_decoder)
            train_loss = criterion(output.view(-1,output.size(-1)),train_target.view(-1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            tpbar.set_description(f'Epoch {epoch+1}/{10}, Loss: {train_loss.item():.4f}')
            writer.add_scalar('train_loss',train_loss.item(),train_global)
            train_global +=1
        model.eval()
        test_L = 0
        with torch.no_grad():
            tpbar = tqdm(test_dataloader)
            for encoder,decoder,target in tpbar:
                encoder = encoder.to(devices)
                decoder = decoder.to(devices)
                target =  target.to(devices)
                output,_ = model(encoder,decoder)
                test_loss = criterion(output.view(-1,output.size(-1)),target.view(-1))
                test_L += test_loss.item()
                tpbar.set_description(f'Epoch {epoch+1}/{10}, Loss: {test_loss.item():4f}')
        writer.add_scalar('test_loss',test_L/len(test_dataloader),test_global)
        test_global +=1

    torch.save(model.state_dict(),'../couplet/model.pth')


