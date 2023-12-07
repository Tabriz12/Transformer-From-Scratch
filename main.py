import torch
from dataset import WMT16
import config
from models.transformer import Transformer
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import data_proc
import tiktoken

tokenizer = tiktoken.get_encoding(config.tokenizer)





def validate_one_epoch(data_loader, model):

    running_loss = 0.

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        en = data[0]
        tr = data[1]
        en_attention = data[2]
        tr_attention = data[3]
        

        out = model(en, en_attention, tr, tr_attention)

        out = out.view(-1, out.size(-1))
        tr = tr.view(-1)

        loss = cross_entropy(out, tr, ignore_index=0) 
        
        running_loss += loss.item()
    

    return running_loss/len(data_loader)

def train_one_epoch(data_loader, model, opt, epoch):
    
    running_loss = 0.
    last_loss = 0.
    for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):
        
        en = data[0]
        tr = data[1]
        en_attention = data[2]
        tr_attention = data[3]

        #en_list = en.tolist()
        #tr_list = tr.tolist()
        #print(tokenizer.decode_batch(en_list))
        #print(tokenizer.decode_batch(tr_list))
        
        opt.zero_grad()

        out = model(en, en_attention, tr, tr_attention)

        '''
        Loss function:

        1. mask the outputs by tr_attention to discard padding tokens in calculation
        2. Calculate loss for each non-padding token in tr and out
        3. Average them out
        
        '''

        out = out.view(-1, out.size(-1))
        tr = tr.view(-1)

        loss = cross_entropy(out, tr, ignore_index=config.pad_token) # ignoring padding tokens which are 0s in calculation

        loss.backward()
        opt.step()

        running_loss += loss.item()

        print( loss.item())

        if i % 500 == 499:
            last_loss = running_loss / 499
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            #for name, param in model.named_parameters():
                #if param.grad is not None:
                   # pass
                    #writer.add_histogram(f'Gradients/{name}', param.grad, global_step=epoch)
            
    #writer.add_scalar('Loss', last_loss, global_step=epoch)

    return last_loss


def main():

    training_data = WMT16(from_disk=True)
    train_loader = data_proc.SmartBatchingDataLoader(training_data, config.batch_size, shuffle=True, collate_fn=data_proc.my_collate_fn, drop_last=True)

    model = Transformer().to(config.device)

    optimizer = Adam(model.parameters(), lr=0.0001)


    for epoch in range(config.epoch):
        model.train()
        train_one_epoch(train_loader, model, optimizer, epoch)


if __name__ == '__main__':
    main()