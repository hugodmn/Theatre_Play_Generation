import torch.nn as nn 
import torch 
import torch.nn.functional as F

class Config():
    def __init__(self, vocab_size, emb_size : int, head_nb : int, block_nb : int, block_size : int, tokenizer_type : str, multi_attn_dropout : float = 0.1, attn_dropout : float = 0.1, head_size :int = 128, train_test_split : float = 0.9):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.head_nb = head_nb
        self.block_nb = block_nb
        self.block_size = block_size
        self.head_size = emb_size // head_nb
        self.attn_dropout = 0.1
        self.multi_attn_dropout = 0.1
        self.tokenizer_type = tokenizer_type
        self.train_test_split = train_test_split


class SingleAttentionHead(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.emb_size = config.emb_size
        self.head_size = config.head_size
        self.block_size = config.block_size
        self.q_linear = nn.Linear(self.emb_size, self.head_size, bias = False)
        self.v_linear = nn.Linear(self.emb_size, self.head_size, bias = False)
        self.k_linear = nn.Linear(self.emb_size, self.head_size, bias = False)

        self.dropout = nn.Dropout(config.attn_dropout)

        # Change
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(self.block_size,self.block_size))
        )
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x) # B, T, C

        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5) # B, T, T

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.v_linear(x)
        out = wei @ v # B, T, C


        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.head_nb = config.head_nb
        self.emb_size = config.emb_size
        self.block_size = config.block_size
        self.heads = nn.ModuleList([SingleAttentionHead(config) for _ in range(self.head_nb)])
        self.proj = nn.Linear(self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(config.multi_attn_dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, config : Config):
        super().__init__()

        self.MHA = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
                nn.Linear(config.emb_size, 4*config.emb_size),
                nn.ReLU(),
                nn.Linear(4*config.emb_size, config.emb_size),
                nn.Dropout(0.1)
        )

        self.ln1 = nn.LayerNorm(config.emb_size)
        self.ln2 = nn.LayerNorm(config.emb_size)

    def forward(self, x):
        out = self.ln1(x)
        out = self.MHA(out)
        out = out + x 
        out = self.ln2(out)
        out = self.feed_forward(out)
        out = out + x
        return out
    
# class FeedForward(nn.Sequential):
#     def __init__(self, config : Config):
#         super().__init__(
#         nn.Linear(config.emb_size, 4*emb_size, bias = False),
#         nn.ReLU(),
#         nn.Linear(4*emb_size, emb_size, bias = False),
#         nn.Dropout(0.1)
#         )
        
class LLM(nn.Module):
    def __init__(self, config : Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.emb_size = config.emb_size
        self.head_nb = config.head_nb
        self.block_nb = config.block_nb
        self.block_size = config.block_size

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = nn.Embedding(self.block_size, config.emb_size)
        #self.blocks = nn.ModuleList([TransformerBlock(self.emb_size, self.head_nb, self.block_size) for _ in range(self.block_nb)])
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(self.block_nb)),
            nn.LayerNorm(config.emb_size)
        )
        # self.ln = nn.LayerNorm(self.emb_size)
  
        self.lm_head = nn.Linear(self.emb_size, config.vocab_size)

    def forward(self, x):
        B,T = x.shape
        
        tok_emb =  self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device = x.device))
        
        out = tok_emb + pos_emb
        # for block in self.blocks:
        #     out = block(out)
        out = self.blocks(out)

        #out = self.ln(out)
        logits = self.lm_head(out)

        return logits
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx