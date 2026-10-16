"""Vectorized greedy/beam decoding across sentences and beams."""
import argparse
from contextlib import nullcontext
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from .tokenizer import PAD, BOS, EOS


@torch.inference_mode()
def batch_decode_ids(model, src, max_length=128, beam_size=1, length_penalty=.6, mixed_precision=False):
    if beam_size < 1:
        raise ValueError('beam_size must be positive')
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.float16) if mixed_precision and src.is_cuda else nullcontext()
    with amp:
        memory, mask = model.encode(src)
        batch, device = src.size(0), src.device
        memory = memory.repeat_interleave(beam_size, 0)
        if mask is not None:
            mask = mask.repeat_interleave(beam_size, 0)
        tokens = torch.full((batch, beam_size, 1), BOS, dtype=torch.long, device=device)
        scores = torch.full((batch, beam_size), -torch.inf, device=device)
        scores[:, 0] = 0
        lengths = torch.zeros(batch, beam_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch, beam_size, dtype=torch.bool, device=device)
        for _ in range(max_length - 1):
            # Project only the last hidden position to the vocabulary during inference.
            logits = model.decode(tokens.reshape(batch * beam_size, -1), memory, mask, last_only=True)[:, -1].float()
            logits[:, [PAD, BOS]] = -torch.inf
            logp = logits.log_softmax(-1).view(batch, beam_size, -1)
            vocab = logp.size(-1)
            if beam_size > vocab - 2:
                raise ValueError('beam_size exceeds available output tokens')
            logp = logp.masked_fill(finished[:, :, None], -torch.inf)
            logp[:, :, PAD] = torch.where(finished, 0., logp[:, :, PAD])
            candidate_scores = scores[:, :, None] + logp
            candidate_lengths = lengths + (~finished).long()
            penalty = ((5 + candidate_lengths.float()) / 6).pow(length_penalty)
            ranks = candidate_scores / penalty[:, :, None]
            _, indices = ranks.flatten(1).topk(beam_size, dim=-1)
            parent, word = indices // vocab, indices % vocab
            tokens = torch.cat((tokens.gather(1, parent[:, :, None].expand(-1, -1, tokens.size(-1))), word[:, :, None]), -1)
            scores = candidate_scores.flatten(1).gather(1, indices)
            lengths = candidate_lengths.gather(1, parent)
            finished = finished.gather(1, parent) | word.eq(EOS)
            if finished.all():
                break
        ranks = scores / ((5 + lengths.float()) / 6).pow(length_penalty)
        best = ranks.argmax(-1)
        return tokens[torch.arange(batch, device=device), best].tolist()


def decode_ids(model, src, max_length=128, beam_size=1, length_penalty=.6):
    if src.size(0) != 1:
        raise ValueError('Use batch_decode_ids for multiple sources')
    return batch_decode_ids(model, src, max_length, beam_size, length_penalty)[0]


def translate_many(model, tokenizer, texts, device, max_length, beam_size=1, length_penalty=.6,
                   batch_size=64, mixed_precision=False, progress=True):
    # Sort by source length and restore original order before reference alignment.
    encoded = [tokenizer.encode(s, max_length) for s in texts]
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
    predictions = [None] * len(texts)
    for start in tqdm(range(0, len(order), batch_size), desc=f'decode beam={beam_size}', disable=not progress, mininterval=1):
        indices = order[start:start + batch_size]
        src = pad_sequence([torch.tensor(encoded[i]) for i in indices], batch_first=True, padding_value=PAD).to(device)
        outputs = batch_decode_ids(model, src, max_length, beam_size, length_penalty, mixed_precision)
        for i, ids in zip(indices, outputs):
            predictions[i] = tokenizer.decode(ids)
    return predictions


def translate(model, tokenizer, text, device, max_length, beam_size=1, length_penalty=.6):
    return translate_many(model, tokenizer, [text], device, max_length, beam_size, length_penalty, progress=False)[0]


if __name__ == '__main__':
    from .evaluate import load_checkpoint
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--beam-size', type=int, default=4)
    a = p.parse_args()
    model, tok, config, _, device = load_checkpoint(a.checkpoint)
    print(translate(model, tok, a.text, device, config['model']['max_length'], a.beam_size))
