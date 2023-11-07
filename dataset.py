import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, block_size : int, split : str ,  to_lower : bool = True):
        self.data = pd.read_csv('datasets/shakespeare-plays/Shakespeare_data.csv')
        self.processed_corpus = []
        self.block_size = block_size
        

        for i, row in self.data.iterrows():
            if to_lower:
                self.processed_corpus.append(row['PlayerLine'].lower())
            else : 
                self.processed_corpus.append(row['PlayerLine'])
        
        if split == 'train':
            self.processed_corpus = self.processed_corpus[:int(0.9*len(self.processed_corpus))]
        elif split == 'test':
            self.processed_corpus = self.processed_corpus[int(0.9*len(self.processed_corpus)):]

        #Merge all the lines into one big string
        self.processed_corpus = '\n'.join(self.processed_corpus)
     
        vocab_chars = set()
        for line in self.processed_corpus:
            for char in line:
                if char not in vocab_chars:
                    vocab_chars.add(char)
        
        self.vocab_size = len(sorted(vocab_chars))
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.vocab_size)
        self.stoi = {char: i for i, char in enumerate(sorted(vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(vocab_chars))}
        # self.encode = lambda x: [stoi[char] for char in x]
        # self.decode = lambda x: [itos[i] for i in x] 

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        return [self.itos[i] for i in x]
    
    def encode(self, x):
        return [int(self.stoi[char]) for char in x]

    def __getitem__(self, idx):
        #print(self.encode(self.processed_corpus[idx : idx + self.block_size + 3]))
        context = self.processed_corpus[idx : idx + self.block_size ]
        prediction = self.processed_corpus[idx + 1 : idx + self.block_size + 1]
        # print("context", context)
        # print("prediction", prediction)
        encoded_context = self.encode(context)
        encoded_prediction = self.encode(prediction)

        encoded_prediction = torch.Tensor(encoded_prediction).type(torch.int32)
  
        #encoded_prediction = self.token_embedding_table(encoded_prediction).squeeze(0)
   
        encoded_context = torch.Tensor(encoded_context).type(torch.int32)
    
        
        return encoded_context, encoded_prediction
    
    def __len__(self):
        return len(self.processed_corpus) - self.block_size - 1 
    


if __name__ == '__main__':
    dataset = ShakespeareDataset(128, 'train')
    print(dataset[0])
    for i,v in dataset.stoi.items():
        print(i,v)
