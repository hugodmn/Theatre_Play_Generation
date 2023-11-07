import torch.nn as nn 
import torch 
import torch.nn.functional as F

class Config():
    def __init__(self, vocab_size : int, emb_size : int, head_nb : int, block_nb : int, block_size : int):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.head_nb = head_nb
        self.block_nb = block_nb
        self.block_size = block_size
        self.head_size = emb_size // head_nb
        self.attn_dropout = 0.1
        self.multi_attn_dropout = 0.1


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