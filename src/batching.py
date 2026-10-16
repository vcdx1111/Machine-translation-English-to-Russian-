"""Deterministic length buckets with a bound on padded source + target tokens."""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
from .tokenizer import PAD


def collate(batch):
    return tuple(pad_sequence([torch.as_tensor(x[i], dtype=torch.long) for x in batch],
                             batch_first=True, padding_value=PAD) for i in (0, 1))


class TokenBatchSampler(Sampler):
    def __init__(self, pairs, max_tokens, max_sentences=512, bucket_size=4096, shuffle=True, seed=42):
        self.lengths = [(len(s), len(t)) for s, t in pairs]
        self.max_tokens, self.max_sentences = max_tokens, max_sentences
        self.bucket_size, self.shuffle, self.seed = bucket_size, shuffle, seed
        if max_tokens < max((s + t for s, t in self.lengths), default=0):
            raise ValueError('max_tokens cannot fit the longest single pair')
        if max_sentences < 1 or bucket_size < 1:
            raise ValueError('max_sentences and bucket_size must be positive')
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for offset in range(0, len(indices), self.bucket_size):
            pool = sorted(indices[offset:offset + self.bucket_size], key=lambda i: (max(self.lengths[i]), self.lengths[i]))
            batch, ms, mt = [], 0, 0
            for i in pool:
                s, t = self.lengths[i]
                ns, nt = max(ms, s), max(mt, t)
                if batch and ((len(batch) + 1) * (ns + nt) > self.max_tokens or len(batch) >= self.max_sentences):
                    batches.append(batch)
                    batch, ms, mt = [], 0, 0
                batch.append(i)
                ms, mt = max(ms, s), max(mt, t)
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def make_loader(rows, tok, cfg, shuffle):
    pairs = [(tok.encode(r['en'], cfg['model']['max_length']),
              tok.encode(r['ru'], cfg['model']['max_length'])) for r in rows]
    training = cfg['training']
    workers = training.get('num_workers', 0)
    use_cuda = training['device'] != 'cpu' and torch.cuda.is_available()
    kwargs = dict(collate_fn=collate, num_workers=workers,
                  pin_memory=training.get('pin_memory', False) and use_cuda,
                  generator=torch.Generator().manual_seed(cfg['seed']))
    if workers:
        # Spawn avoids forking a process after CUDA/CPU thread pools were initialized.
        # Token IDs stay Python lists so workers do not share hundreds of thousands of tensor FDs.
        kwargs.update(persistent_workers=True, prefetch_factor=training.get('prefetch_factor', 2),
                      multiprocessing_context='spawn')
    if training.get('max_tokens'):
        sampler = TokenBatchSampler(pairs, training['max_tokens'], training.get('max_sentences', 512),
                                    training.get('bucket_size', 4096), shuffle, cfg['seed'])
        return DataLoader(pairs, batch_sampler=sampler, **kwargs)
    return DataLoader(pairs, batch_size=training['batch_size'], shuffle=shuffle, **kwargs)


def set_loader_epoch(loader, epoch, seed):
    if isinstance(loader.batch_sampler, TokenBatchSampler):
        loader.batch_sampler.set_epoch(epoch)
    loader.generator.manual_seed(seed + epoch)


def loader_stats(loader):
    pairs = loader.dataset
    actual, padded = 0, 0
    for batch in loader.batch_sampler:
        lengths = [(len(pairs[i][0]), len(pairs[i][1])) for i in batch]
        actual += sum(s + t for s, t in lengths)
        padded += len(batch) * (max(s for s, _ in lengths) + max(t for _, t in lengths))
    return dict(batches=len(loader), pairs=len(pairs), nonpad_tokens=actual, padded_tokens=padded,
                padding_fraction=1 - actual / padded, target_tokens=sum(len(t) - 1 for _, t in pairs))
