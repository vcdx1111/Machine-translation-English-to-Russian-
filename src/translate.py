"""Autoregressive decoding; beam width one is greedy decoding."""
import argparse
import torch
from .tokenizer import PAD, BOS, EOS


@torch.inference_mode()
def decode_ids(model, src, max_length=128, beam_size=1, length_penalty=.6):
    if beam_size < 1:
        raise ValueError('beam_size must be positive')
    model.eval()
    memory, mask = model.encode(src)
    if src.size(0) != 1:
        raise ValueError('Decoding expects one source sentence')
    beams = [([BOS], 0.0)]
    def rank(item):
        tokens, score = item
        return score / (((5 + len(tokens) - 1) / 6) ** length_penalty)
    for _ in range(max_length - 1):
        candidates = []
        for tokens, score in beams:
            if tokens[-1] == EOS:
                candidates.append((tokens, score))
                continue
            tgt = torch.tensor([tokens], device=src.device)
            logits = model.decode(tgt, memory, mask)[0, -1].float()
            logits[[PAD, BOS]] = -torch.inf
            logp = logits.log_softmax(-1)
            values, indices = logp.topk(min(beam_size, logp.numel() - 2))
            candidates.extend((tokens + [i], score + v) for v, i in zip(values.tolist(), indices.tolist()))
        beams = sorted(candidates, key=rank, reverse=True)[:beam_size]
        if all(tokens[-1] == EOS for tokens, _ in beams):
            break
    return max(beams, key=rank)[0]


def translate(model, tokenizer, text, device, max_length, beam_size=1, length_penalty=.6):
    src = torch.tensor([tokenizer.encode(text, max_length)], device=device)
    return tokenizer.decode(decode_ids(model, src, max_length, beam_size, length_penalty))


if __name__ == '__main__':
    from .evaluate import load_checkpoint
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--beam-size', type=int, default=4)
    a = p.parse_args()
    model, tok, config, _, device = load_checkpoint(a.checkpoint)
    print(translate(model, tok, a.text, device, config['model']['max_length'], a.beam_size))
