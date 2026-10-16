"""Post-norm Transformer with explicit scaled dot-product multi-head attention."""
import math
import torch
from torch import nn
from .tokenizer import PAD


def source_mask(tokens):
    return (tokens != PAD)[:, None, None, :]


def target_mask(tokens):
    n = tokens.size(1)
    return source_mask(tokens) & torch.ones(n, n, dtype=torch.bool, device=tokens.device).tril()[None, None]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        self.heads, self.dk = heads, d_model // heads
        self.q, self.k, self.v, self.out = [nn.Linear(d_model, d_model) for _ in range(4)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        b = query.size(0)
        def project(layer, x):
            return layer(x).view(b, -1, self.heads, self.dk).transpose(1, 2)
        q, k, v = project(self.q, query), project(self.k, key), project(self.v, value)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)
        attended = self.dropout(weights) @ v
        return self.out(attended.transpose(1, 2).contiguous().view(b, -1, self.heads * self.dk))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_length):
        super().__init__()
        if d_model % 2:
            raise ValueError('d_model must be even')
        p = torch.arange(max_length).float()[:, None]
        rates = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(p * rates), torch.cos(p * rates)
        self.register_buffer('pe', pe[None])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError('Sequence exceeds configured positional encoding')
        return self.dropout(x + self.pe[:, :x.size(1)].to(x.dtype))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.drop(self.ff(x)))


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, heads=8, layers=6, d_ff=2048, dropout=.1, max_length=128):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model, dropout, max_length)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)])
        self.output = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def encode(self, src):
        x = self.position(self.embedding(src) * self.scale)
        mask = source_mask(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return x, mask

    def decode(self, tgt, memory, src_mask):
        x = self.position(self.embedding(tgt) * self.scale)
        mask = target_mask(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, mask)
        return self.output(x)

    def forward(self, src, tgt):
        memory, mask = self.encode(src)
        return self.decode(tgt, memory, mask)
