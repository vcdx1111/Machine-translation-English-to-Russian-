"""Batched corpus evaluation; full greedy scores and a separately labelled beam subset."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import torch
from sacrebleu.metrics import BLEU, CHRF
from .data import load_rows
from .model import Transformer
from .tokenizer import Tokenizer
from .translate import translate_many


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    cfg = state['config']
    import sentencepiece as spm
    tok = Tokenizer.__new__(Tokenizer)
    tok.sp = spm.SentencePieceProcessor(model_proto=state['tokenizer_model'])
    torch.set_num_threads(cfg['training'].get('cpu_threads', 2))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training']['device'] != 'cpu' else 'cpu')
    model = Transformer(len(tok), **cfg['model']).to(device)
    model.load_state_dict(state['best_model'])
    return model, tok, cfg, state, device


def reference_groups(rows):
    references = defaultdict(list)
    for r in rows:
        if r['ru'] not in references[r['en']]:
            references[r['en']].append(r['ru'])
    return references


def score_predictions(predictions, sources, references):
    refs = [[references[s][i] if i < len(references[s]) else None for s in sources]
            for i in range(max(len(references[s]) for s in sources))]
    bleu, chrf = BLEU(tokenize='13a'), CHRF()
    b, c = bleu.corpus_score(predictions, refs), chrf.corpus_score(predictions, refs)
    return dict(bleu=b.score, chrf=c.score, bleu_signature=str(bleu.get_signature()), chrf_signature=str(chrf.get_signature()))


def decode_sources(model, tok, cfg, sources, device, beam_size=1, batch_size=None):
    ev = cfg['evaluation']
    return translate_many(model, tok, sources, device, cfg['model']['max_length'], beam_size,
                          ev['length_penalty'], batch_size or ev.get('batch_size', 64),
                          cfg['training']['mixed_precision'], cfg['training'].get('progress', True))


def validation_greedy(model, tok, cfg, rows, device):
    # A fixed, bounded validation source subset; never beam-search during training.
    refs = reference_groups(rows)
    n = cfg['evaluation'].get('val_greedy_samples', 0)
    sources = list(refs)[:n]
    if not sources:
        return dict(val_greedy_bleu=None, val_greedy_chrf=None, val_greedy_seconds=0., val_greedy_sources=0)
    begin = time.perf_counter()
    pred = decode_sources(model, tok, cfg, sources, device, 1)
    score = score_predictions(pred, sources, refs)
    return dict(val_greedy_bleu=score['bleu'], val_greedy_chrf=score['chrf'],
                val_greedy_seconds=time.perf_counter() - begin, val_greedy_sources=len(sources))


def evaluate_model(model, tok, cfg, rows, output, device, training_seconds, data_hash):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = reference_groups(rows)
    sources = list(references)
    if not sources:
        raise ValueError('Empty test split')
    begin = time.perf_counter()
    ev = cfg['evaluation']
    beam = ev['beam_size']
    limit = ev.get('beam_max_sources')
    if beam < 1 or (limit is not None and limit < 0):
        raise ValueError('beam_size must be positive and beam_max_sources must be nonnegative')
    # No limit: backward-compatible full beam evaluation. A limit: full greedy + extra subset beam.
    full_beam = beam > 1 and (limit is None or limit >= len(sources))
    primary_beam = beam if full_beam else 1
    predictions = decode_sources(model, tok, cfg, sources, device, primary_beam)
    score = score_predictions(predictions, sources, references)
    greedy_seconds = time.perf_counter() - begin
    metrics = dict(test_bleu=score['bleu'], test_chrf=score['chrf'],
                   bleu_signature=score['bleu_signature'], chrf_signature=score['chrf_signature'],
                   test_decoding='beam' if full_beam else 'greedy', test_beam_size=primary_beam,
                   training_seconds=training_seconds, test_sources=len(sources), test_pairs=len(rows),
                   primary_evaluation_seconds=greedy_seconds, config=cfg, data_sha256=data_hash, device=str(device),
                   purpose='cpu_smoke_only' if cfg['data'].get('max_train') else 'full_training')
    if beam > 1 and not full_beam and limit:
        # Hash ordering fixes the subset independent of length, predictions, or model quality.
        subset = sorted(sources, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:limit]
        started = time.perf_counter()
        beam_pred = decode_sources(model, tok, cfg, subset, device, beam, ev.get('beam_batch_size', 32))
        beam_score = score_predictions(beam_pred, subset, references)
        by_source = dict(zip(sources, predictions))
        subset_greedy = score_predictions([by_source[s] for s in subset], subset, references)
        metrics.update(beam_subset_sources=len(subset), beam_subset_size=beam,
                       beam_subset_bleu=beam_score['bleu'], beam_subset_chrf=beam_score['chrf'],
                       beam_subset_bleu_signature=beam_score['bleu_signature'],
                       beam_subset_chrf_signature=beam_score['chrf_signature'],
                       beam_subset_greedy_bleu=subset_greedy['bleu'], beam_subset_greedy_chrf=subset_greedy['chrf'],
                       beam_subset_seconds=time.perf_counter() - started)
    metrics['evaluation_seconds'] = time.perf_counter() - begin
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'samples.txt').open('w', encoding='utf-8') as f:
        for source, hypothesis in list(zip(sources, predictions))[:30]:
            f.write(f'EN: {source}\nREF: {" || ".join(references[source])}\nMODEL: {hypothesis}\n\n')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--results-dir')
    p.add_argument('--beam-size', type=int)
    p.add_argument('--beam-max-sources', type=int, help='Override subset size; use a large value for full beam')
    a = p.parse_args()
    model, tok, cfg, state, device = load_checkpoint(a.checkpoint)
    if a.beam_size is not None:
        cfg['evaluation']['beam_size'] = a.beam_size
    if a.beam_max_sources is not None:
        cfg['evaluation']['beam_max_sources'] = a.beam_max_sources
    if hashlib.sha256(Path(cfg['paths']['raw']).read_bytes()).hexdigest() != state['data_sha256']:
        raise ValueError('Raw dataset differs from checkpoint')
    from .data import prepare
    prepare(cfg['paths']['raw'], cfg['paths']['processed'], cfg['seed'])
    rows = load_rows(cfg['paths']['processed'], 'test')
    import random
    random.Random(cfg['seed']).shuffle(rows)
    limit = cfg['data'].get('max_test')
    if limit:
        rows = rows[:limit]
    print(json.dumps(evaluate_model(model, tok, cfg, rows, a.results_dir or cfg['paths']['results'],
                                   device, state['training_seconds'], state['data_sha256']), ensure_ascii=False))
