"""CPU throughput and full-data batching audit; never a substitute for a T4 measurement."""
import argparse
from itertools import islice
import json
from pathlib import Path
import platform
import random
import time
import torch
import yaml
import numpy as np
from .batching import make_loader, loader_stats, collate
from .data import prepare, load_rows
from .model import Transformer
from .tokenizer import train_tokenizer, PAD
from .train import seed_all, Noam, run_epoch


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/transformer_colab_fast.yaml')
    p.add_argument('--output', default='results/cpu_throughput.json')
    p.add_argument('--steps', type=int, default=12)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['training'].get('cpu_threads', 2))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    prepare(cfg['paths']['raw'], cfg['paths']['processed'], cfg['seed'])
    rows = load_rows(cfg['paths']['processed'], 'train')
    random.Random(cfg['seed']).shuffle(rows)
    tok = train_tokenizer(rows, out.parent / 'benchmark_tokenizer', cfg['tokenizer']['vocab_size'])
    lengths = {lang: np.array([len(tok.sp.encode(r[lang])) + 2 for r in rows]) for lang in ('en', 'ru')}
    length_stats = {lang: {str(p): float(np.percentile(v, p)) for p in (50, 90, 95, 99, 100)} for lang, v in lengths.items()}
    cap = cfg['model']['max_length']
    clipped = np.logical_or(lengths['en'] > cap, lengths['ru'] > cap)
    full_loader = make_loader(rows, tok, cfg, True)
    full_stats = loader_stats(full_loader)
    # Isolate padding-policy savings under the SAME vocabulary and cap.
    old = yaml.safe_load(Path(a.config).read_text())
    old['training'].pop('max_tokens', None)
    old['training']['batch_size'] = 48
    fixed_stats = loader_stats(make_loader(rows, tok, old, True))
    cfg['training'].update(device='cpu', mixed_precision=False, max_tokens=2048, max_sentences=64, num_workers=0, pin_memory=False)
    loader = make_loader(rows[:4096], tok, cfg, True)
    # Materialize a bounded set outside timing; training measurements include model/optimizer work.
    batches = list(islice(loader, a.steps + 2))
    model = Transformer(len(tok), **cfg['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(.9, .98), eps=1e-9)
    schedule = Noam(optimizer, cfg['model']['d_model'], cfg['training']['warmup_steps'], cfg['training'].get('lr_factor', 1.0))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg['training']['label_smoothing'])
    run_epoch(model, batches[:2], criterion, torch.device('cpu'), optimizer, schedule, scaler)
    stats = {}
    run_epoch(model, batches[2:], criterion, torch.device('cpu'), optimizer, schedule, scaler, stats=stats)
    epochs = cfg['training']['epochs']
    # Explicit scenarios rather than asserting a CPU-to-GPU conversion ratio.
    scenarios = {}
    for multiplier in (5, 10, 20):
        tps = stats['target_tokens_per_second'] * multiplier
        train_seconds = full_stats['target_tokens'] * epochs / tps
        scenarios[str(multiplier)] = dict(assumed_T4_vs_CPU_token_rate=multiplier, required_T4_target_tokens_per_second=tps,
                                           train_only_hours=train_seconds / 3600,
                                           total_hours_with_20percent_val_io_and_15min_test=train_seconds * 1.2 / 3600 + .25)
    result = dict(config=a.config, python=platform.python_version(), torch=torch.__version__,
                  architecture=cfg['model'], vocabulary=len(tok), parameters=sum(p.numel() for p in model.parameters()),
                  cpu_threads=torch.get_num_threads(), cpu_batch_max_tokens=2048, cpu_max_sentences=64,
                  train_length_percentiles_including_bos_eos=length_stats,
                  pair_max_p99=float(np.percentile(np.maximum(lengths['en'], lengths['ru']),99)),
                  cap=cap, truncated_train_pairs=int(clipped.sum()), truncated_train_fraction=float(clipped.mean()),
                  full_fast_batches=full_stats, random_48_same_vocab_and_cap=fixed_stats,
                  measured_cpu_training=stats, scenario_estimates=scenarios,
                  target_T4_target_tokens_per_second_for_2h=full_stats['target_tokens'] * epochs * 1.2 / ((2 - .25) * 3600),
                  target_T4_target_tokens_per_second_for_3h=full_stats['target_tokens'] * epochs * 1.2 / ((3 - .25) * 3600),
                  caveat='CPU measurements are real; T4 multipliers and overheads are unmeasured assumptions. Use one measured T4 epoch and runtime_estimate.json instead.')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
