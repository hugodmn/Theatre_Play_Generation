import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.model import Config
import string
from tqdm import tqdm 


class PreprocessData():
    def __init__(self, config : Config):
        self.to_lower = True 
        self.char_token_bool = False
        self.tokenizer_type = config.tokenizer_type
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train_test_split = config.train_test_split

    def clean_a_bit(self, corpus):
        clean_corpus = []
        clean_scene = []
        punct_after = ''
        print(len(corpus))
        with tqdm(range(len(corpus))) as pbar :
            for scene in corpus : 
                for word in scene : 
                    if len(word)>1 :   
                        if word[-1] in string.punctuation :
                            if len(word)>2 :
                                if word[-2] in string.punctuation and word[-3] in string.punctuation : 
                                    word = word[:-2]
                            punct_after = word[-1]
                            word = word[:-1]
                                
            
                        if word[-2:] == "'d" and len(word) != 2:
                            word = word[:-2] + "ed"
                    clean_scene.append(word)
                    if punct_after != '':
                        clean_scene.append(punct_after)
                    punct_after = ''
                clean_corpus.append(clean_scene)
                pbar.update(1)
        
        return clean_corpus
            

    def tokenize(self, corpus : pd.DataFrame):
        
        self.processed_corpus = []

        all_chars = set()
        entire_scene = []
        scene_started : bool = False

        for i, row in corpus.iterrows():

            
            if isinstance(row['Player'],  str) == True :
                if row['Player'] != last_player :
                    if self.to_lower : 
                        entire_scene.append(str(row['Player']).lower()+' :')
                    else :
                        entire_scene.append(str(row['Player'])+' :')
                last_player = row['Player']
            else : 
                last_player = None

                
            if row['Player'] not in all_chars and isinstance(row['Player'],  str) == True:
                all_chars.add(row['Player'])

            if self.to_lower:
                entire_scene.append(row['PlayerLine'].lower())
            else : 
                entire_scene.append(row['PlayerLine'])

            if (row['PlayerLine'][0:5] == 'SCENE') or i == len(corpus) - 1:
                if not scene_started :
                    scene_started = True
                else :
                    
                    if self.char_token_bool : 
                        entire_scene = '<CHAR'+str(len(all_chars))+'>'+' \n '+' \n '.join(entire_scene)+' \n '
                    else :
                        entire_scene = ' \n '.join(entire_scene)+' \n '
                    entire_scene = entire_scene.split(' ')
                    self.processed_corpus.append(entire_scene)
                    entire_scene = []
                    all_chars = set()




        self.train_processed_corpus = self.processed_corpus[:int(self.train_test_split*len(self.processed_corpus))]
        self.test_processed_corpus = self.processed_corpus[int(self.train_test_split*len(self.processed_corpus)):]

        self.train_processed_corpus = self.clean_a_bit(self.train_processed_corpus)
        self.test_processed_corpus = self.clean_a_bit(self.test_processed_corpus)

        if self.tokenizer_type == 'char_level' :

            vocab_chars = set()


            for corpus in [self.train_processed_corpus, self.test_processed_corpus]:
                for scene in corpus :
                    for word in scene:
                        if self.char_token_bool :
                            if word[:5] == '<CHAR':
                                if word not in vocab_chars:
                                    vocab_chars.add(word)
                            else :
                                for char in word :
                                    if char not in vocab_chars:
                                        vocab_chars.add(char)
                        else :
                            for char in word :
                                if char not in vocab_chars:
                                    vocab_chars.add(char)
            #adding space to the vocab 
            vocab_chars.add(' ')


            self.vocab_size = len(sorted(vocab_chars))
            self.stoi = {char: i for i, char in enumerate(sorted(vocab_chars))}
            self.itos = {i: char for i, char in enumerate(sorted(vocab_chars))}

            self.train_processed_corpus = [x for subset in self.train_processed_corpus for x in subset]
            self.train_processed_corpus = ' '.join(self.train_processed_corpus)

            self.train_processed_encoded_corpus = []
            i=0
            while i < len(self.train_processed_corpus):
                if self.train_processed_corpus[i] == '<' and self.char_token_bool :

                    if self.train_processed_corpus[i+7] == '>' :
                        self.train_processed_encoded_corpus.append(self.stoi[self.train_processed_corpus[i:i+8]])
                        i+=8
                    elif self.train_processed_corpus[i+6] == '>' :
                        self.train_processed_encoded_corpus.append(self.stoi[self.train_processed_corpus[i:i+7]])
                        i+=7
                else : 
                    self.train_processed_encoded_corpus.append(self.stoi[self.train_processed_corpus[i]])
                    i+=1

            self.test_processed_corpus = [x for subset in self.test_processed_corpus for x in subset]
            self.test_processed_corpus = ' '.join(self.test_processed_corpus)

            self.test_processed_encoded_corpus = []
            j=0
            
            while j < len(self.test_processed_corpus):
                if self.test_processed_corpus[j] == '<' and self.char_token_bool :
                    if self.test_processed_corpus[j+7] == '>' :
                        self.test_processed_encoded_corpus.append(self.stoi[self.test_processed_corpus[j:j+8]])
                        j+=8
          
                    elif self.test_processed_corpus[j+6] == '>' :
                        self.test_processed_encoded_corpus.append(self.stoi[self.test_processed_corpus[j:j+7]])
                        j+=7
                else : 
                    self.test_processed_encoded_corpus.append(self.stoi[self.test_processed_corpus[j]])
                    j+=1
                #print(j)

            self.test_processed_corpus = self.test_processed_encoded_corpus
            self.train_processed_corpus = self.train_processed_encoded_corpus
           

        elif self.tokenizer_type == 'word_level' :
            vocab_chars = set()

            

            for corpus in [self.train_processed_corpus, self.test_processed_corpus]:
                for scene in corpus:
                    #scene = scene.split(' ')
                    for word in scene :
                        if word not in vocab_chars:
                            vocab_chars.add(word)
                #vocab_chars.add('\n')


            self.vocab_size = len(sorted(vocab_chars))
            self.stoi = {char: i for i, char in enumerate(sorted(vocab_chars))}
            self.itos = {i: char for i, char in enumerate(sorted(vocab_chars))}

            self.train_processed_corpus_list = [x for subset in self.train_processed_corpus for x in subset]
            self.train_processed_corpus = [self.stoi[x] for x in self.train_processed_corpus_list]


            self.test_processed_corpus_list = [x for subset in self.test_processed_corpus for x in subset]
            self.test_processed_corpus = [self.stoi[x] for x in self.test_processed_corpus_list]


        elif self.tokenizer_type == 'bert_tokenizer' :


            
            self.train_processed_corpus = [x for subset in self.train_processed_corpus for x in subset]
            self.train_processed_corpus = ''.join(self.train_processed_corpus)

            self.test_processed_corpus = [x for subset in self.test_processed_corpus for x in subset]
            self.test_processed_corpus = ''.join(self.test_processed_corpus)
           
            self.train_processed_corpus = self.tokenizer(self.train_processed_corpus, add_special_tokens=False).input_ids
            self.test_processed_corpus = self.tokenizer(self.test_processed_corpus, add_special_tokens=False).input_ids

            self.vocab_size = self.tokenizer.vocab_size


        return self.train_processed_corpus, self.test_processed_corpus




class ShakespeareDataset(Dataset):
    def __init__(self, config : Config, split : str ):
        self.data = pd.read_csv('datasets/shakespeare-plays/Shakespeare_data.csv')
        self.config = config
        self.block_size = self.config.block_size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


        self.PreprocessData = PreprocessData(config)
        self.train_processed_corpus, self.test_processed_corpus = self.PreprocessData.tokenize(self.data)
        
        self.itos = self.PreprocessData.itos
        self.stoi = self.PreprocessData.stoi
        
        if split == 'train':
            self.processed_corpus = self.train_processed_corpus
        elif split == 'test':
            self.processed_corpus = self.test_processed_corpus

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()

        if self.config.tokenizer_type == 'bert_tokenizer' :

            return self.tokenizer.convert_ids_to_tokens(x)
        
        elif self.config.tokenizer_type == 'char_level' :
            return [self.itos[i] for i in x]
        
        elif self.config.tokenizer_type == 'word_level' :
            return [self.itos[i] for i in x]
    
    def encode(self, x, char_level_tokenizer : bool = False, word_level_tokenizer : bool = False):

        if self.config.tokenizer_type == 'bert_tokenizer' :
            return self.tokenizer(x).input_ids
        
        elif self.config.tokenizer_type == 'char_level' :

            return [int(self.stoi[char]) for char in x]
        
        elif self.config.tokenizer_type == 'word_level' :

            return [int(self.stoi[word]) for word in x.split(' ')]

    def __getitem__(self, idx):
        #print(self.encode(self.processed_corpus[idx : idx + self.block_size + 3]))

        context = self.processed_corpus[idx : idx + self.block_size ]
        prediction = self.processed_corpus[idx + 1 : idx + self.block_size + 1]
    
        prediction = torch.Tensor(prediction).type(torch.int32)
        context = torch.Tensor(context).type(torch.int32)

        return context, prediction
    
    def __len__(self):

        return len(self.processed_corpus) - self.block_size - 1 
    


if __name__ == '__main__':

    config = Config(emb_size = 512, 
                    head_nb = 4, 
                    block_nb = 4, 
                    block_size = 32, 
                    tokenizer_type = 'word_level', 
                    train_test_split = 0.8)
    dataset = ShakespeareDataset('train', config)
    print(dataset[0])
    # for i,v in dataset.stoi.items():
    #     print(i,v)
