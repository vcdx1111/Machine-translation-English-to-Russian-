"""Test-set BLEU/chrF with reproducible signatures and multi-reference scoring."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from sacrebleu.metrics import BLEU, CHRF
from .data import load_rows
from .model import Transformer
from .tokenizer import Tokenizer
from .translate import translate


def load_checkpoint(path):
    # Only load checkpoints you created/trust: optimizer/RNG resume state uses pickle.
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


def evaluate_model(model, tok, cfg, rows, output, device, training_seconds, data_hash):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    references = defaultdict(list)
    for r in rows:
        if r['ru'] not in references[r['en']]:
            references[r['en']].append(r['ru'])
    sources = list(references)
    if not sources:
        raise ValueError('Empty test split')
    begin = time.perf_counter()
    predictions = [translate(model, tok, source, device, cfg['model']['max_length'],
                            cfg['evaluation']['beam_size'], cfg['evaluation']['length_penalty']) for source in sources]
    # sacreBLEU accepts None for missing references at a particular position.
    refs = [[references[s][i] if i < len(references[s]) else None for s in sources]
            for i in range(max(map(len, references.values()))) ]
    bleu, chrf = BLEU(tokenize='13a'), CHRF()
    b, c = bleu.corpus_score(predictions, refs), chrf.corpus_score(predictions, refs)
    metrics = dict(test_bleu=b.score, test_chrf=c.score, bleu_signature=str(bleu.get_signature()),
                   chrf_signature=str(chrf.get_signature()), training_seconds=training_seconds,
                   evaluation_seconds=time.perf_counter() - begin, test_sources=len(sources), test_pairs=len(rows),
                   config=cfg, data_sha256=data_hash, device=str(device),
                   purpose='cpu_smoke_only' if cfg['data'].get('max_train') else 'full_training')
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
    a = p.parse_args()
    model, tok, cfg, state, device = load_checkpoint(a.checkpoint)
    if a.beam_size:
        cfg['evaluation']['beam_size'] = a.beam_size
    # Standalone evaluation must target the same dataset used in training.
    import hashlib
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
