import torch 
import torch.nn as nn 
from tqdm import tqdm 
#from dataset import ShakespeareDataset
from dataset import ShakespeareDataset
from torch.utils.data import DataLoader, Subset
from models.model import LLM, Config
import torch.nn.functional as F
import json

def step(model, optimizer, scheduler, train_loader, test_dataset,  device, epoch, path, best_loss, result_dict = None):
    model.train()
    total_acc = 0
    total_loss = 0
    sample_nb = 0
    loss_for_n_steps = 0

    with tqdm(range(len(train_loader))) as pbar :
        for idx, (context,targets) in enumerate(train_loader):
            context, targets = context.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(context)
            B,T,C = logits.shape
            #B,T = targets.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
            loss.backward()

            optimizer.step()
            if idx % 200 == 0:
                print(f'loss for step {idx} : {loss_for_n_steps/200}')
                
                test_loss, best_loss = test_for_n_steps(model, test_dataset, device, path, best_loss, epoch, 200)
                result_dict['steps'].append(idx)
                result_dict['test_loss'].append(test_loss)
                result_dict['train_loss'].append(loss_for_n_steps/200)
                loss_for_n_steps = 0

                # input = test_dataset.tokenizer('this is not the best of hyperbilious times, th')['input_ids']
                input = test_dataset.encode(" ")
                #input = input.split(' ')
                input = torch.Tensor(input).type(torch.int32).to(device).unsqueeze(0)
                output = model.generate(input, 500).squeeze(0).cpu()
                decoded_output = test_dataset.decode(output)

                if test_dataset.config.tokenizer_type == 'char_level' :
                    decoded_output = ''.join(decoded_output)
                print(decoded_output)
                # final_output = ""

            if idx % 3000 == 0 :
                #save to json result_dict
                with open(path+'result_dict.json', 'w') as f:
                    json.dump(result_dict, f, indent=2)
                scheduler.step()
              
            total_acc += (logits.argmax(-1) == targets).sum().item()
            pbar.update(1)
            
            sample_nb += B
            total_loss += loss.item()
            loss_for_n_steps += loss.item()


    print(f'[TRAIN EPOCH {epoch}] Accuracy : {total_acc*100/(sample_nb*T)}% Train Loss : {total_loss/len(train_loader)}')

def test_for_n_steps(model, test_dataset, device, path, best_loss, epoch, n_steps, batch_size=32):
        # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(test_dataset))[:n_steps*batch_size]
    subset_dataset = Subset(test_dataset, subset_indices)

    # Create the subset DataLoader
    subset_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    with torch.no_grad():
        total_acc_test = 0 
        total_loss_test = 0
        sample_nb_test = 0
        step = 0
        with tqdm(range(n_steps)) as pbar :
            for idx,(context, targets) in enumerate(subset_dataloader):
                #context, targets = next(iter(test_loader))
                context, targets = context.to(device), targets.to(device)
                logits = model(context)
                B,T,C = logits.shape
                #B,T = targets.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

                sample_nb_test += B

                total_acc_test += (logits.argmax(-1) == targets).sum().item()
                total_loss_test += loss.item()

                step += 1
                pbar.update(1)
            if total_loss_test/n_steps < best_loss :
                best_loss = total_loss_test/n_steps
                torch.save(model.state_dict(), path+'model_[T:'+str(model.config.tokenizer_type)+'].pt')
        print(f'[TEST EPOCH {epoch}] Accuracy : {total_acc_test*100/(sample_nb_test*T)}% Test Loss : {total_loss_test/n_steps} Best Loss : {best_loss}')
        return best_loss
    


if __name__ == '__main__' : 
    epochs = 20 
    device = 'cuda'
    print_all_vocab = False
    block_size = 256 # -> context length 

    emb_size = 512
    head_nb = 4
    block_nb = 6
    LLM_config = Config(
        vocab_size = "unknown_for_now",
        emb_size = emb_size,
        head_nb = head_nb,
        block_nb = block_nb,
        block_size = block_size,
        tokenizer_type = 'word_level',
        train_test_split = 0.9)
    
    result_dict = {
        'steps' : [],
        'text_generated' : [],
        'train_loss' : [],
        'test_loss' : []
    }

    train_dataset = ShakespeareDataset(LLM_config, 'train')
    test_dataset = ShakespeareDataset(LLM_config, 'test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    vocab_size = len(train_dataset.stoi)
    LLM_config.vocab_size = vocab_size
    best_loss = 10000
    print(f'Vocab size : {vocab_size}')

    #vocab_size, emb_size : int, head_nb : int, block_nb : int, block_size : int, tokenizer_type : str, train_test_split : float = 0.9):

    if print_all_vocab : 
        print('-----------------------Printing all vocab-----------------------')
        for tok in train_dataset.stoi : 
            print(tok)
        print('-----------------------End of vocab-----------------------')
    


    model = LLM(LLM_config).to(device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)
    #Save config attributes in a json file
    if LLM_config.tokenizer_type == 'bert_tokenizer' :
        path = 'models/bert_tokens/'
    elif LLM_config.tokenizer_type == 'char_level' :
        path = 'models/char_tokens/'
    elif LLM_config.tokenizer_type == 'word_level' :
        path = 'models/word_tokens/'

    with open(path+'config.json', 'w') as f:
        json.dump(LLM_config.__dict__, f, indent=2)

    with open(path+'stio.json', 'w') as f:
        json.dump(train_dataset.stoi, f, indent=2)

    with open(path+'itos.json', 'w') as f:
        json.dump(train_dataset.itos, f, indent=2)

    
    print('---------------------Starting training---------------------')
    for epoch in range(epochs):
        step(model, optimizer, scheduler, train_loader, test_dataset, device, epoch, path, best_loss = 1000)
        #best_loss = test(model, test_loader, device, best_loss, epoch)
