import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, block_size : int, split : str ,  to_lower : bool = True):
        self.data = pd.read_csv('datasets/shakespeare-plays/Shakespeare_data.csv')
        self.processed_corpus = []
        self.block_size = block_size
        

        entire_scene = []
        all_chars = set()
        last_player = None
        scene_started : bool = False

        for i, row in self.data.iterrows():

            
            if isinstance(row['Player'],  str) == True :
                if row['Player'] != last_player :
                    entire_scene.append(str(row['Player'])+':')
                last_player = row['Player']
            else : 
                last_player = None

                
            if row['Player'] not in all_chars and row['Player'] != '' :
                all_chars.add(row['Player'])

            if to_lower:
                entire_scene.append(row['PlayerLine'].lower())
            else : 
                entire_scene.append(row['PlayerLine'])

            if (row['PlayerLine'][0:5] == 'SCENE') or i == len(self.data) - 1:
                if not scene_started :
                    scene_started = True
                else :
                    
                    # self.processed_corpus.append('<CHAR'+str(len(all_chars))+'>')
                    self.processed_corpus.append('<CHAR'+str(len(all_chars))+'>'+'\n'+'\n'.join(entire_scene))
                    entire_scene = []
                    all_chars = set()

        
        if split == 'train':
            self.processed_corpus = self.processed_corpus[:int(0.9*len(self.processed_corpus))]
        elif split == 'test':
            self.processed_corpus = self.processed_corpus[int(0.9*len(self.processed_corpus)):]

        
        #Merge all the lines into one big string
        #self.processed_corpus = '\n'.join(self.processed_corpus)
     
        vocab_chars = set()
        
        for line in self.processed_corpus:
            for idx in range(len(line)):
                #print(line)
                if line[idx] not in vocab_chars and line[idx] != '<':
                    vocab_chars.add(line[idx])
                if line[idx] == '<':
                    #print(idx)
                    #print(line[idx:idx+7])
                    if line[idx+6] == '>':
                  
                        vocab_chars.add(line[idx:idx+7])
                        idx += 7
                    else : 
                        vocab_chars.add(line[idx:idx+8])
                        idx += 8
                    
                    
        
        self.vocab_size = len(sorted(vocab_chars))
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.vocab_size)
        self.stoi = {char: i for i, char in enumerate(sorted(vocab_chars))}
        self.itos = {i: char for i, char in enumerate(sorted(vocab_chars))}
 


    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        return [self.itos[i] for i in x]
    
    def encode(self, x):
        encoded_x = []
        idx = 0
        print(x[0])
        while idx < len(x):
            if x[idx] == '<':
                if x[idx+6] == '>':
                  
                    encoded_x.append([int(self.stoi[x[idx:idx+7]])])
                    idx += 7
                else : 
                    encoded_x.append([int(self.stoi[x[idx:idx+8]])])
                    idx += 8
            else :
                # print('x[idx] : ', x[idx])
                # print('stoi : ', self.stoi)
                encoded_x.append([int(self.stoi[x[idx]])])
                idx += 1
            
        return encoded_x

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
        #print(encoded_prediction)

        encoded_context = torch.Tensor(encoded_context).type(torch.int32)
    
        
        return encoded_context, encoded_prediction
    
    def __len__(self):
        return len(self.processed_corpus) - self.block_size - 1 
    


if __name__ == '__main__':
    dataset = ShakespeareDataset(128, 'train')
    print(dataset[0])
   
