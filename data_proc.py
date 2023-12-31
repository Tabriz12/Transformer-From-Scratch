import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import random
from torch.utils.data import DataLoader
import config



def finer_batches(indices, batch_size):

    '''

    This function provides randomness in batched data to prevent overfitting
    
    '''
    
    tot = len(indices)

    not_selected = []

    for _ in tqdm(range(0, (tot//batch_size) * batch_size, batch_size), total=tot // batch_size ):

        start_idx = random.randint(0, len(indices)-1)

        start_idx = min(start_idx, len(indices)- batch_size)

        selected_ids = indices[start_idx:start_idx+batch_size]

        not_selected.extend(selected_ids)

        del indices[start_idx:start_idx+batch_size]
    
    
    return not_selected


class SmartBatchingDataLoader(DataLoader):

    '''
    
    Smart Batching is a technique to optimize training by batching samples with similar size

    
    '''
    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None, drop_last=False):

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)

        self.shuffle = shuffle

    def __iter__(self):
        if not self.shuffle:
            # If not shuffling, use the default iterator
            return super().__iter__()

        # Shuffle the indices while considering the sequence lengths
        indices = self.dataset.get_indices_sorted_by_length()

        indices_shuffled = finer_batches(indices, self.batch_size) 
        

        for i in range(0, len(indices_shuffled), self.batch_size):

            batch_indices = indices_shuffled[i:i + self.batch_size]

            # Fetch actual data samples corresponding to the indices
            batch_data = [self.dataset[idx] for idx in batch_indices]

            # If collate function is provided, use it to collate the batch
            if self.collate_fn is not None:
                batch_data = self.collate_fn(batch_data)

            yield batch_data


def attention_mask(sequence, max_len):

    seq_len = len(sequence)

    line = torch.Tensor([
        [0.0] * seq_len + [float('-inf')] * (max_len - seq_len)
    ]).squeeze()

    top = line.unsqueeze(0).repeat(seq_len, 1)
    bottom = torch.full(((max_len-seq_len), max_len), float('-inf'))
    mask_tensor = torch.cat((top, bottom), dim=0).unsqueeze(0)
    return mask_tensor



def my_collate_fn_old(batch):

    en = [sentence[0] for sentence in batch]
    tr =[sentence[1] for sentence in batch]

    # Pad sequences to the length of the longest sequence in the batch
    en_padded = pad_sequence(en, batch_first=True)
    tr_padded = pad_sequence(tr, batch_first=True)

    
    en_max = en_padded.size(dim=1)

    tr_max = tr_padded.size(dim=1)
  
    for idx, sequence in enumerate(en):
       if not idx: 
           en_attention_mask = attention_mask(sequence, en_max)
       else:
           en_attention_mask = torch.cat((en_attention_mask,  attention_mask(sequence, en_max)), dim=0)
       

    for idx, sequence in enumerate(tr):
       if not idx: 
           tr_attention_mask = attention_mask(sequence, tr_max)
       else:
           tr_attention_mask = torch.cat((tr_attention_mask,  attention_mask(sequence, tr_max)), dim=0)

    return en_padded, tr_padded, en_attention_mask.to(config.device), tr_attention_mask.to(config.device)

def my_collate_fn(batch):

    en = [sentence[0] for sentence in batch]
    tr =[sentence[1] for sentence in batch]


    # Pad sequences to the length of the longest sequence in the batch
    en_padded = pad_sequence(en, batch_first=True, padding_value=config.pad_token)
    tr_padded = pad_sequence(tr, batch_first=True, padding_value=config.pad_token)
    
    en_max = en_padded.size(dim=1)

    tr_max = tr_padded.size(dim=1)


    en_attention_mask = []
    for sequence in en:
        seq_len = len(sequence)
        en_attention_mask.append([1] * seq_len + [0] * (en_max - seq_len))
    
    tr_attention_mask = []
    for sequence in tr:
        seq_len = len(sequence)
        tr_attention_mask.append([1] * seq_len + [0] * (tr_max - seq_len))

    

    return en_padded, tr_padded, en_attention_mask, tr_attention_mask



def find_index(list, el):

    for idx, item in enumerate(list):

        if item == el:
            return idx


def random_sorted_batches(indices, batch_size, select_window):

    '''
    More randomized batches but too slow, may optimize
    '''

    tot = len(indices)

    not_selected = []

    for _ in tqdm(range(0, tot, batch_size), total=int(tot/batch_size)):
    
        start_item = random.choice(indices)
        
        cur_id = find_index(indices, start_item)

        left = min(cur_id, select_window)  # selecting window size from left

        right = min(len(indices) - 1 - cur_id, select_window) # selecting window size from right

        selected_ids = random.sample(indices[cur_id-left:cur_id+right], batch_size)

        not_selected.append(selected_ids)

        indices = list(set(indices).difference(set(selected_ids)))

    
    return not_selected