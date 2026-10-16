"""Configurable pre/post-norm Transformer with explicit scaled dot-product multi-head attention."""
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
    def __init__(self, d_model, heads, d_ff, dropout, norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.norm_first:
            z = self.norm1(x)
            x = x + self.drop(self.attn(z, z, z, mask))
            return x + self.drop(self.ff(self.norm2(x)))
        x = self.norm1(x + self.drop(self.attn(x, x, x, mask)))
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        if self.norm_first:
            z = self.norm1(x)
            x = x + self.drop(self.self_attn(z, z, z, tgt_mask))
            x = x + self.drop(self.cross_attn(self.norm2(x), memory, memory, src_mask))
            return x + self.drop(self.ff(self.norm3(x)))
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.drop(self.ff(x)))


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, heads=8, layers=6, d_ff=2048, dropout=.1, max_length=128,
                 norm_first=False, tie_embeddings=False, initialization="legacy_xavier"):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = PositionalEncoding(d_model, dropout, max_length)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout, norm_first) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout, norm_first) for _ in range(layers)])
        self.encoder_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.decoder_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.output = nn.Linear(d_model, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if initialization not in ('legacy_xavier', 'scaled_embedding'):
            raise ValueError('Unknown initialization')
        if initialization == 'scaled_embedding':
            # After sqrt(d_model) scaling, content has unit variance independent of vocabulary size.
            nn.init.normal_(self.embedding.weight, mean=0., std=d_model ** -.5)
            for module in self.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        if tie_embeddings:
            self.output.weight = self.embedding.weight
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def encode(self, src):
        x = self.position(self.embedding(src) * self.scale)
        mask = source_mask(src)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.encoder_norm(x), mask

    def decode(self, tgt, memory, src_mask, last_only=False):
        x = self.position(self.embedding(tgt) * self.scale)
        mask = target_mask(tgt)
        for layer in self.decoder:
            x = layer(x, memory, src_mask, mask)
        return self.output(self.decoder_norm(x[:, -1:] if last_only else x))

    def forward(self, src, tgt):
        memory, mask = self.encode(src)
        return self.decode(tgt, memory, mask)
