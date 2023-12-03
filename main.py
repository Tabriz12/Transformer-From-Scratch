import torch
from dataset import WMT16
import config
from models.transformer import Transformer
from torch.optim import SparseAdam
from tqdm import tqdm
import my_utils


def train_one_epoch(data_loader, model, opt, epoch):
    
    running_loss = 0.
    last_loss = 0.
    for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):

        #print(data[0].size(), data[1].size())
        
        en = data[0]
        tr = data[1]
        en_attention = data[2]
        tr_attention = data[3]
        
        opt.zero_grad()

        loss = model(en, en_attention, tr, tr_attention)

        loss.backward()

        ss
        # Adjust learning weights
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
    train_loader = my_utils.SmartBatchingDataLoader(training_data, config.batch_size, shuffle=True, collate_fn=my_utils.my_collate_fn, drop_last=True)

    model = Transformer().to(config.device)

    optimizer = SparseAdam(model.parameters(), lr=0.0001)


    for epoch in range(config.epoch):
        model.train()
        train_one_epoch(train_loader, model, optimizer, epoch)


if __name__ == '__main__':
    main()