"""Reproducible training, complete epoch-boundary resume, and final evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import yaml
from .data import prepare, load_rows
from .tokenizer import PAD, train_tokenizer, Tokenizer
from .model import Transformer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch):
    return tuple(pad_sequence([torch.tensor(x[i]) for x in batch], batch_first=True, padding_value=PAD) for i in (0, 1))


def make_loader(rows, tok, cfg, shuffle):
    pairs = [(tok.encode(r['en'], cfg['model']['max_length']), tok.encode(r['ru'], cfg['model']['max_length'])) for r in rows]
    return DataLoader(pairs, batch_size=cfg['training']['batch_size'], shuffle=shuffle, collate_fn=collate, num_workers=0)


class Noam:
    def __init__(self, optimizer, d_model, warmup, factor=1.0):
        self.optimizer, self.d_model, self.warmup, self.factor = optimizer, d_model, warmup, factor
        self.steps = 0

    def step(self):
        self.steps += 1
        lr = self.factor * self.d_model ** -.5 * min(self.steps ** -.5, self.steps * self.warmup ** -1.5)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda' and scaler is not None and scaler.is_enabled())):
                logits = model(src, tgt[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                # AMP may overflow before scaler calibration: skip that update and schedule step.
                if torch.isfinite(norm):
                    scheduler.step()
                elif not scaler.is_enabled():
                    raise RuntimeError('Nonfinite gradients')
                scaler.step(optimizer)
                scaler.update()
            tokens = (tgt[:, 1:] != PAD).sum().item()
            total += loss.item() * tokens
            count += tokens
    if not count:
        raise ValueError('Empty epoch')
    return total / count


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def run(config_path, resume=None, checkpoint_dir=None, results_dir=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if checkpoint_dir:
        cfg['paths']['checkpoints'] = checkpoint_dir
    if results_dir:
        cfg['paths']['results'] = results_dir
    seed_all(cfg['seed'])
    torch.set_num_threads(cfg['training'].get('cpu_threads', 2))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training']['device'] != 'cpu' else 'cpu')
    if cfg['training']['device'] == 'cuda' and device.type != 'cuda':
        raise RuntimeError('CUDA requested but unavailable; use transformer_cpu.yaml for smoke tests')
    paths = cfg['paths']
    checkpoint_root, output = Path(paths['checkpoints']), Path(paths['results'])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite an earlier experiment or carry its best checkpoint forward.
    if not resume and any((checkpoint_root / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Existing experiment: use --resume or a new --checkpoint-dir')
    data_hash = hashlib.sha256(Path(paths['raw']).read_bytes()).hexdigest()
    state = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if state:
        old = state['config']
        for key in ('seed', 'model', 'tokenizer', 'data'):
            if old[key] != cfg[key]:
                raise ValueError(f'Cannot resume with changed {key}')
        for key in cfg['training']:
            if key not in ('epochs', 'device', 'cpu_threads') and old['training'][key] != cfg['training'][key]:
                raise ValueError(f'Cannot resume with changed training.{key}')
        if state['data_sha256'] != data_hash:
            raise ValueError('Raw data changed since checkpoint')
    prepare(paths['raw'], paths['processed'], cfg['seed'])
    rows = {name: load_rows(paths['processed'], name) for name in ('train', 'val', 'test')}
    # Sample independently AFTER full grouped splitting; no test data fits the tokenizer.
    for name, values in rows.items():
        random.Random(cfg['seed']).shuffle(values)
        limit = cfg['data'].get(f'max_{name}')
        rows[name] = values[:limit] if limit else values
    if state:
        tokenizer_path = checkpoint_root / 'tokenizer.model'
        tokenizer_path.write_bytes(state['tokenizer_model'])
        tok = Tokenizer(tokenizer_path)
    else:
        tok = train_tokenizer(rows['train'], checkpoint_root / 'tokenizer', cfg['tokenizer']['vocab_size'])
    tokenizer_bytes = (checkpoint_root / 'tokenizer.model').read_bytes()
    model = Transformer(len(tok), **cfg['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(.9, .98), eps=1e-9)
    scheduler = Noam(optimizer, cfg['model']['d_model'], cfg['training']['warmup_steps'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['training']['mixed_precision'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg['training']['label_smoothing'])
    start, best, bad, history, elapsed = 0, float('inf'), 0, [], 0.0
    best_weights = None
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        scheduler.steps = state['scheduler_steps']
        start, best, bad = state['epoch'], state['best_loss'], state['bad_epochs']
        history, elapsed, best_weights = state['history'], state['training_seconds'], state['best_model']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        if device.type == 'cuda' and state['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
    train_loader = make_loader(rows['train'], tok, cfg, True)
    val_loader = make_loader(rows['val'], tok, cfg, False)
    log_path = output / 'train_log.csv'
    def write_log():
        with log_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr', 'epoch_seconds'])
            writer.writeheader()
            writer.writerows(history)
    write_log()
    for epoch in range(start, cfg['training']['epochs']):
        if bad >= cfg['training']['patience']:
            break
        begin = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, cfg['training']['gradient_clip'])
        val_loss = run_epoch(model, val_loader, criterion, device)
        duration = time.perf_counter() - begin
        elapsed += duration
        improved = val_loss < best - cfg['training']['min_delta']
        if improved:
            best, bad = val_loss, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                            lr=optimizer.param_groups[0]['lr'], epoch_seconds=duration))
        write_log()
        state = dict(config=cfg, model=model.state_dict(), best_model=best_weights, optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), scheduler_steps=scheduler.steps, epoch=epoch + 1,
                     best_loss=best, bad_epochs=bad, history=history, training_seconds=elapsed,
                     tokenizer_model=tokenizer_bytes, data_sha256=data_hash,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        save_checkpoint(checkpoint_root / 'last.pt', state)
        if improved:
            save_checkpoint(checkpoint_root / 'best.pt', state)
        print(json.dumps(history[-1]), flush=True)
    if best_weights is None:
        raise ValueError('No completed training epoch')
    # A portable resumed last.pt carries its best weights even if best.pt was not copied.
    model.load_state_dict(best_weights)
    from .evaluate import evaluate_model
    metrics = evaluate_model(model, tok, cfg, rows['test'], output, device, elapsed, data_hash)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/transformer_base.yaml')
    p.add_argument('--resume', help='Trusted local last.pt; PyTorch checkpoints use pickle')
    p.add_argument('--checkpoint-dir')
    p.add_argument('--results-dir')
    args = p.parse_args()
    run(args.config, args.resume, args.checkpoint_dir, args.results_dir)
