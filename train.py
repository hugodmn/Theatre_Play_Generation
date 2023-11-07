import torch 
import torch.nn as nn 
from tqdm import tqdm 
from dataset import ShakespeareDataset
from torch.utils.data import DataLoader
from model import LLM, Config
import torch.nn.functional as F

def step(model, optimizer, train_loader, test_loader,  device, epoch, best_loss):
    model.train()
    total_acc = 0
    total_loss = 0
    sample_nb = 0

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
            if idx % 1000 == 0:
                print(f'loss for step {idx} : {loss.item()}')
                best_loss = test_for_n_steps(model, test_loader, device, best_loss, epoch, 500)

            total_acc += (logits.argmax(-1) == targets).sum().item()
            pbar.update(1)
            
            sample_nb += B
            total_loss += loss.item()


    print(f'[TRAIN EPOCH {epoch}] Accuracy : {total_acc*100/(sample_nb*T)}% Train Loss : {total_loss/len(train_loader)}')

def test_for_n_steps(model, test_loader, device, best_loss, epoch, n_steps):
    model.eval()
    with torch.no_grad():
        total_acc_test = 0 
        total_loss_test = 0
        sample_nb_test = 0
        step = 0
        with tqdm(range(n_steps)) as pbar :
            while step < n_steps:
                context, targets = next(iter(test_loader))
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
            if total_loss_test < best_loss :
                best_loss = total_loss_test
                torch.save(model.state_dict(), 'model.pt')
        print(f'[TEST EPOCH {epoch}] Accuracy : {total_acc_test*100/(sample_nb_test*T)}% Test Loss : {total_loss_test/n_steps} Best Loss : {best_loss}')
        return best_loss
    



def test(model, test_loader, device, best_loss, epoch):
    model.eval()
    with torch.no_grad():
        total_acc = 0 
        total_loss = 0
        sample_nb = 0
        with tqdm(range(len(test_loader))) as pbar :
            for idx, (context, targets) in enumerate(test_loader):
                context, targets = context.to(device), targets.to(device)
                logits = model(context)
                B,T,C = logits.shape
                #B,T = targets.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

                sample_nb += B

                total_acc += (logits.argmax(-1) == targets).sum().item()
                total_loss += loss.item()

                # if idx % 100 == 0:
                #     print(loss.item())
                pbar.update(1)
            if total_loss < best_loss :
                best_loss = total_loss
                torch.save(model.state_dict(), 'model.pt')
        print(f'[TEST EPOCH {epoch}] Accuracy : {total_acc*100/(sample_nb*T)}% Test Loss : {total_loss/len(test_loader)} Best Loss : {best_loss}')
        return best_loss

if __name__ == '__main__' : 
    epochs = 20 
    device = 'mps'

    block_size = 128 # -> context length 
    train_dataset = ShakespeareDataset(block_size, 'train')
    test_dataset = ShakespeareDataset(block_size, 'test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    best_loss = 10000
   

    vocab_size = train_dataset.vocab_size
    emb_size = 256
    head_nb = 8
    block_nb = 6
    LLM_config = Config(vocab_size, emb_size, head_nb, block_nb, block_size)
    model = LLM(LLM_config).to(device) 
    # for idx, (context,targets) in enumerate(train_loader):
    #     if idx == 3 : 
    #         print(context.shape)
    #         print(context)
    #         print(targets.shape)
    #         print(targets)
    #         preds = model(context.to(device))
    #         print(preds.shape)
    #         print(preds)
        

    # 
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        step(model, optimizer, train_loader, test_loader, device, epoch, best_loss = 1000)
        #best_loss = test(model, test_loader, device, best_loss, epoch)
