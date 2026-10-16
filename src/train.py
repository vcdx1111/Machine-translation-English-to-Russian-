"""Token-bucket training with measured progress, AMP checks and epoch-boundary resume."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml
from .batching import collate, make_loader, set_loader_epoch, loader_stats
from .data import selected_splits
from .tokenizer import PAD, train_tokenizer, Tokenizer
from .model import Transformer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


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


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, clip=1.0,
              stats=None, progress=False, description='epoch', mixed_precision=None, log_every=25):
    training = optimizer is not None
    model.train(training)
    amp_enabled = device.type == 'cuda' and (scaler is not None and scaler.is_enabled() if mixed_precision is None else mixed_precision)
    loss_sum = torch.zeros((), device=device)
    count, padded, all_tokens, skipped = 0, 0, 0, 0
    first_dtype = None
    sync(device)
    begin = time.perf_counter()
    bar = tqdm(loader, desc=description, disable=not progress, mininterval=1, unit='step')
    with torch.set_grad_enabled(training):
        for step, (src, tgt) in enumerate(bar, 1):
            # Host counts avoid an extra device synchronization on every batch.
            tokens = (tgt[:, 1:] != PAD).sum().item()
            all_tokens += (src != PAD).sum().item() + (tgt != PAD).sum().item()
            padded += src.numel() + tgt.numel()
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(src, tgt[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            if first_dtype is None:
                first_dtype = str(logits.dtype)
                if amp_enabled and logits.dtype != torch.float16:
                    raise RuntimeError(f'FP16 AMP requested but logits dtype is {logits.dtype}')
                if progress:
                    tqdm.write(f'{description}: autocast={amp_enabled}, logits={logits.dtype}, '
                               f'GradScaler={scaler.is_enabled() if scaler is not None else False}')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(norm):
                    scheduler.step()
                elif not scaler.is_enabled():
                    raise RuntimeError('Nonfinite gradients')
                else:
                    skipped += 1
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.detach().float() * tokens
            count += tokens
            if progress and (step == 1 or step % log_every == 0 or step == len(loader)):
                # Read loss only at log intervals; tqdm displays step ETA and throughput.
                bar.set_postfix(loss=f'{loss_sum.item()/count:.3f}', target_tok_s=f'{count/(time.perf_counter()-begin):.0f}',
                                lr=f'{optimizer.param_groups[0]["lr"]:.2g}' if training else '-', skipped=skipped)
    sync(device)
    seconds = time.perf_counter() - begin
    if not count:
        raise ValueError('Empty epoch')
    if stats is not None:
        stats.update(seconds=seconds, target_tokens=count, target_tokens_per_second=count / seconds,
                     nonpad_tokens=all_tokens, nonpad_tokens_per_second=all_tokens / seconds,
                     padding_fraction=1 - all_tokens / padded, steps=len(loader), skipped_steps=skipped,
                     amp_enabled=amp_enabled, logits_dtype=first_dtype)
    return loss_sum.item() / count


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def runtime_estimate(history, cfg, test_sources):
    recent = history[-min(3, len(history)):]
    per_epoch = sum(r['epoch_seconds'] + r.get('checkpoint_seconds', 0) for r in recent) / len(recent)
    greedy_sources = sum(r.get('val_greedy_sources', 0) for r in recent)
    greedy_seconds = sum(r.get('val_greedy_seconds', 0.) for r in recent)
    ev = cfg['evaluation']
    rate = greedy_sources / greedy_seconds if greedy_seconds > 0 else None
    limit = ev.get('beam_max_sources')
    beam_sources = min(test_sources, test_sources if limit is None else limit) if ev['beam_size'] > 1 else 0
    full_beam = ev['beam_size'] > 1 and (limit is None or limit >= test_sources)
    equivalent_greedy_sources = (test_sources * ev['beam_size'] if full_beam
                                 else test_sources + beam_sources * ev['beam_size'])
    if full_beam and ev.get('report_greedy', False):
        equivalent_greedy_sources += test_sources
    if ev.get('repetition_ablation_ngram', 0):
        equivalent_greedy_sources += test_sources * ev['beam_size']
    # A planning assumption, NOT a measured beam throughput; final evaluation records actual timings.
    decode_seconds = equivalent_greedy_sources / rate if rate else None
    remaining_epochs = max(0, cfg['training']['epochs'] - history[-1]['epoch'])
    total = per_epoch * cfg['training']['epochs'] + (decode_seconds or 0)
    return dict(measured_epoch_seconds=per_epoch, completed_epoch=history[-1]['epoch'],
                planned_epochs=cfg['training']['epochs'], remaining_train_seconds=per_epoch * remaining_epochs,
                projected_total_hours=total / 3600, projected_test_seconds=decode_seconds,
                greedy_sources_per_second=rate, beam_time_assumption='beam_size times measured validation greedy cost per source',
                note='Estimate excludes one-time setup; data distribution, decoding lengths and Drive speed can change it. Early stopping can shorten it.',
                within_3_hour_target=total <= 3 * 3600)


def run(config_path, resume=None, checkpoint_dir=None, results_dir=None, stop_after_epochs=None, skip_test=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if checkpoint_dir:
        cfg['paths']['checkpoints'] = checkpoint_dir
    if results_dir:
        cfg['paths']['results'] = results_dir
    seed_all(cfg['seed'])
    training = cfg['training']
    torch.set_num_threads(training.get('cpu_threads', 2))
    device = torch.device('cuda' if torch.cuda.is_available() and training['device'] != 'cpu' else 'cpu')
    if training['device'] == 'cuda' and device.type != 'cuda':
        raise RuntimeError('CUDA requested but unavailable; use a CPU config for smoke tests')
    paths = cfg['paths']
    checkpoint_root, output = Path(paths['checkpoints']), Path(paths['results'])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any((checkpoint_root / name).exists() for name in ('last.pt', 'best.pt')):
        raise FileExistsError('Existing experiment: use --resume or a new --checkpoint-dir')
    data_hash = hashlib.sha256(Path(paths['raw']).read_bytes()).hexdigest()
    state = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if state:
        old = state['config']
        for key in ('seed', 'model', 'tokenizer', 'data'):
            if old[key] != cfg[key]:
                raise ValueError(f'Cannot resume with changed {key}; start a new fast experiment')
        runtime_keys = ('epochs', 'device', 'cpu_threads', 'num_workers', 'pin_memory', 'prefetch_factor', 'progress', 'log_every')
        for key in training:
            if key not in runtime_keys and old['training'].get(key) != training[key]:
                raise ValueError(f'Cannot resume with changed training.{key}')
        if state.get('pipeline_version', 1) != 2:
            raise ValueError('This batching/RNG pipeline changed; start a new run instead of resuming legacy checkpoints')
        if state['data_sha256'] != data_hash:
            raise ValueError('Raw data changed since checkpoint')
    print(f'Preparing {paths["raw"]}', flush=True)
    rows = selected_splits(cfg)
    if state:
        tokenizer_path = checkpoint_root / 'tokenizer.model'
        tokenizer_path.write_bytes(state['tokenizer_model'])
        tok = Tokenizer(tokenizer_path)
    else:
        print('Fitting train-only SentencePiece', flush=True)
        tok = train_tokenizer(rows['train'], checkpoint_root / 'tokenizer', cfg['tokenizer']['vocab_size'])
    tokenizer_bytes = (checkpoint_root / 'tokenizer.model').read_bytes()
    model = Transformer(len(tok), **cfg['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(.9, .98), eps=1e-9)
    scheduler = Noam(optimizer, cfg['model']['d_model'], training['warmup_steps'], training.get('lr_factor', 1.0))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and training['mixed_precision'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=training['label_smoothing'])
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
    test_sources = len({r['en'] for r in rows['test']})
    print(json.dumps(dict(device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                          amp_requested=training['mixed_precision'], amp_enabled=scaler.is_enabled(),
                          parameters=sum(p.numel() for p in model.parameters()),
                          num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
                          train_batches=loader_stats(train_loader)), indent=2), flush=True)
    log_path = output / 'train_log.csv'
    def write_log():
        columns = ['epoch', 'train_loss', 'val_loss', 'lr', 'epoch_seconds', 'checkpoint_seconds',
                   'train_seconds', 'val_seconds', 'target_tokens_per_second', 'nonpad_tokens_per_second',
                   'padding_fraction', 'val_greedy_bleu', 'val_greedy_chrf', 'val_greedy_seconds',
                   'val_greedy_sources', 'cuda_peak_gb', 'amp_enabled', 'logits_dtype', 'skipped_steps']
        with log_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(history)
    write_log()
    limit_epoch = min(training['epochs'], start + stop_after_epochs) if stop_after_epochs else training['epochs']
    for epoch in range(start, limit_epoch):
        if bad >= training['patience']:
            break
        set_loader_epoch(train_loader, epoch, cfg['seed'])
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        train_stats, val_stats = {}, {}
        common = dict(progress=training.get('progress', True), mixed_precision=scaler.is_enabled(),
                      log_every=training.get('log_every', 25))
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler,
                               training['gradient_clip'], stats=train_stats, description=f'train {epoch+1}/{training["epochs"]}', **common)
        val_loss = run_epoch(model, val_loader, criterion, device, stats=val_stats,
                             description=f'val {epoch+1}/{training["epochs"]}', **common)
        from .evaluate import validation_greedy
        val_greedy = validation_greedy(model, tok, cfg, rows['val'], device)
        improved = val_loss < best - training['min_delta']
        if improved:
            best, bad = val_loss, 0
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        duration = time.perf_counter() - begin
        elapsed += duration
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                            lr=optimizer.param_groups[0]['lr'], epoch_seconds=duration, checkpoint_seconds=0.,
                            train_seconds=train_stats['seconds'], val_seconds=val_stats['seconds'],
                            target_tokens_per_second=train_stats['target_tokens_per_second'],
                            nonpad_tokens_per_second=train_stats['nonpad_tokens_per_second'],
                            padding_fraction=train_stats['padding_fraction'], **val_greedy,
                            cuda_peak_gb=torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else 0.,
                            amp_enabled=train_stats['amp_enabled'], logits_dtype=train_stats['logits_dtype'],
                            skipped_steps=train_stats['skipped_steps']))
        state = dict(pipeline_version=2, config=cfg, model=model.state_dict(), best_model=best_weights,
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), scheduler_steps=scheduler.steps,
                     epoch=epoch + 1, best_loss=best, bad_epochs=bad, history=history, training_seconds=elapsed,
                     tokenizer_model=tokenizer_bytes, data_sha256=data_hash,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        checkpoint_begin = time.perf_counter()
        save_checkpoint(checkpoint_root / 'last.pt', state)
        if improved:
            save_checkpoint(checkpoint_root / 'best.pt', state)
        history[-1]['checkpoint_seconds'] = time.perf_counter() - checkpoint_begin
        write_log()
        estimate = runtime_estimate(history, cfg, test_sources)
        (output / 'runtime_estimate.json').write_text(json.dumps(estimate, indent=2))
        print(json.dumps(history[-1]), flush=True)
        print(f'Epoch {epoch+1}: {duration:.1f}s + checkpoint {history[-1]["checkpoint_seconds"]:.1f}s; '
              f'train ETA {estimate["remaining_train_seconds"]/60:.1f} min; '
              f'projected train+test {estimate["projected_total_hours"]:.2f} h', flush=True)
        if not estimate['within_3_hour_target']:
            print('WARNING: measured first/recent epochs exceed the 3h target; see runtime_estimate.json. '
                  'A new smaller experiment or fewer planned epochs may be needed.', flush=True)
    if best_weights is None:
        raise ValueError('No completed training epoch')
    if skip_test:
        print('Calibration complete; test skipped. Resume last.pt to continue the SAME run.', flush=True)
        return runtime_estimate(history, cfg, test_sources)
    model.load_state_dict(best_weights)
    from .evaluate import evaluate_model
    metrics = evaluate_model(model, tok, cfg, rows['test'], output, device, elapsed, data_hash)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/transformer_colab_fast.yaml')
    p.add_argument('--resume', help='Trusted local last.pt; checkpoints use pickle')
    p.add_argument('--checkpoint-dir')
    p.add_argument('--results-dir')
    p.add_argument('--stop-after-epochs', type=int, help='Run this many additional epochs without changing the planned config')
    p.add_argument('--skip-test', action='store_true', help='Calibration: save checkpoint/ETA but do not touch test predictions')
    args = p.parse_args()
    run(args.config, args.resume, args.checkpoint_dir, args.results_dir, args.stop_after_epochs, args.skip_test)
