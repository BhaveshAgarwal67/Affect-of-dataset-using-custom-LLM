import torch
import torch.nn as nn
from torch.nn import functional as f

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = f.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = scores @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, block_size, dropout)
        self.mlp = MultiLayerPerceptron(d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Transformer(d_model, n_head, block_size, dropout) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(d_model)
        self.de_embd = nn.Linear(d_model, vocab_size)
        self.apply(self.weights)

    def weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, 'bias', None) is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedded = self.token_embedding_table(idx)
        position_embedded = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_embedded + position_embedded
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.de_embd(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = f.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_tokens, temperature=1.0, top_k=5):
        for _ in range(max_tokens):
            idx_cropped = idx[:, -self.block_size:]
            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            prob = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx