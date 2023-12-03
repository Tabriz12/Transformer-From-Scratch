import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import config
import tiktoken
import pickle
import os

class WMT16(Dataset):
    def __init__(self, 
                 from_disk=True,
                 dataset = config.dataset,
                 subset = config.subset,
                 split = 'train',
                 tokenizer = config.tokenizer,
                 cache_file='tokenized_dataset.pkl'):
        super().__init__()

        self.tokenizer = tiktoken.get_encoding(tokenizer)

        #print((self.tokenizer.n_vocab)) #vocab size
 
        if from_disk: ## Not to tokenize dataset every time

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.dataset = pickle.load(f)
            else:
                raise ValueError(f"Tokenized dataset not found at {cache_file}. Tokenize the dataset first by setting 'from_disk' = False")

        else:
            self.dataset = load_dataset(dataset, subset, split=split)
        
            self.dataset = [[self.tokenizer.encode(sentence['translation']['en']), self.tokenizer.encode(sentence['translation']['tr'])] for sentence in self.dataset]
            
            with open(cache_file, 'wb') as f:
                    pickle.dump(self.dataset, f)

    

    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, index):

        #implement embedding + padding for each batch (+ attention masks)

        en = torch.LongTensor(self.dataset[index][0]).to(config.device)
        tr = torch.LongTensor(self.dataset[index][1]).to(config.device)

        #print(self.tokenizer.decode(self.dataset[index][0]))
        #print(self.tokenizer.decode(self.dataset[index][1]))

        return en, tr
    
    def get_indices_sorted_by_length(self):
        # Return indices sorted by the length of sequences
        indices = list(range(len(self)))
        indices.sort(key=lambda x: len(self[x][0]))
        return indices



#[7553, 8143, 25438, 129268, 132928, 144660, 149875, 165566, 165591, 23236]