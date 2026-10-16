"""Same-split Marian zero-shot and Seq2SeqTrainer fine-tuning. Run GPU learning in Colab."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import yaml
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from .data import selected_splits
from .evaluate import reference_groups, score_predictions
from .results import test_identity, write_artifacts

LOG_FIELDS = ['epoch', 'train_loss', 'val_loss', 'lr', 'epoch_seconds', 'global_step', 'cuda_peak_gb']


def write_log(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ParallelDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        for start in range(0, len(rows), 1024):
            batch = rows[start:start + 1024]
            encoded = tokenizer([r['en'] for r in batch], text_target=[r['ru'] for r in batch],
                                max_length=max_length, truncation=True)
            self.items.extend({k: v[i] for k, v in encoded.items()} for i in range(len(batch)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class MarianCollator:
    """Dynamic padding + CPU label shift, without pickling a CUDA model into workers."""
    def __init__(self, tokenizer, config, multiple=None):
        self.padding = DataCollatorForSeq2Seq(tokenizer, model=None, pad_to_multiple_of=multiple)
        self.pad, self.start = config.pad_token_id, config.decoder_start_token_id

    def __call__(self, features):
        batch = self.padding(features)
        labels = batch['labels']
        decoder = torch.full_like(labels, self.pad)
        decoder[:, 0] = self.start
        decoder[:, 1:] = labels[:, :-1]
        decoder.masked_fill_(decoder == -100, self.pad)
        batch['decoder_input_ids'] = decoder
        return batch


@torch.inference_mode()
def generate_texts(model, tokenizer, cfg, sources, device, beams):
    model.eval()
    ev = cfg['evaluation']
    size = ev['beam_batch_size'] if beams > 1 else ev['batch_size']
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    predictions = [None] * len(sources)
    amp = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' and cfg['training']['mixed_precision'] else nullcontext()
    with amp:
        for start in tqdm(range(0, len(order), size), desc=f'Marian beam={beams}',
                          disable=not cfg['training']['progress'], mininterval=1):
            indices = order[start:start + size]
            inputs = tokenizer([sources[i] for i in indices], padding=True, truncation=True,
                               max_length=cfg['data']['max_length'], return_tensors='pt').to(device)
            output = model.generate(**inputs, num_beams=beams, do_sample=False,
                                    max_new_tokens=ev['max_new_tokens'], length_penalty=ev['length_penalty'] if beams > 1 else 1.0,
                                    no_repeat_ngram_size=ev.get('no_repeat_ngram_size', 0),
                                    early_stopping=(beams > 1), use_cache=True)
            for i, text in zip(indices, tokenizer.batch_decode(output, skip_special_tokens=True)):
                predictions[i] = text
    return predictions


def evaluate(model, tokenizer, cfg, rows, output, stage, identity, revision, training_seconds):
    begin = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    refs = reference_groups(rows)
    sources = list(refs)
    device = next(model.parameters()).device
    greedy = generate_texts(model, tokenizer, cfg, sources, device, 1)
    gs = score_predictions(greedy, sources, refs)
    beams = cfg['evaluation']['beam_size']
    predictions = generate_texts(model, tokenizer, cfg, sources, device, beams) if beams > 1 else greedy
    bs = score_predictions(predictions, sources, refs)
    metrics = dict(**identity, test_bleu=bs['bleu'], test_chrf=bs['chrf'],
                   bleu_signature=bs['bleu_signature'], chrf_signature=bs['chrf_signature'],
                   test_greedy_bleu=gs['bleu'], test_greedy_chrf=gs['chrf'],
                   test_beam_bleu=bs['bleu'], test_beam_chrf=bs['chrf'],
                   test_decoding='beam' if beams > 1 else 'greedy', test_beam_size=beams,
                   model_family='marian_cpu_fixture' if revision.startswith('local:') else 'helsinki_marian',
                   pretrained_model_validated=not revision.startswith('local:'), stage=stage, base_revision=revision,
                   no_repeat_ngram_size=cfg['evaluation'].get('no_repeat_ngram_size', 0),
                   config=cfg, device=str(device), training_seconds=training_seconds,
                   evaluation_seconds=time.perf_counter() - begin,
                   purpose='cpu_smoke_only' if cfg['data'].get('max_train') else stage,
                   pretraining_overlap_risk='OPUS may include Tatoeba; test overlap has not been ruled out')
    write_artifacts(output, metrics, sources, refs, predictions)
    if stage == 'zero_shot':
        write_log(output / 'train_log.csv', [])
    return metrics


class MarianTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        loader = super().get_train_dataloader()
        if self.args.dataloader_num_workers:
            getattr(loader, 'base_dataloader', loader).multiprocessing_context = 'spawn'
        return loader

    def get_eval_dataloader(self, eval_dataset=None):
        loader = super().get_eval_dataloader(eval_dataset)
        if self.args.dataloader_num_workers:
            getattr(loader, 'base_dataloader', loader).multiprocessing_context = 'spawn'
        return loader


class EpochLog(TrainerCallback):
    """Live epoch summaries and durable CSV/ETA across reconnects; Trainer owns optimizer/RNG state."""
    def __init__(self, output, planned_epochs, stop_after_epochs=None):
        self.output = Path(output)
        self.planned_epochs, self.stop_after = planned_epochs, stop_after_epochs
        self.rows = []
        path = self.output / 'train_log.csv'
        if path.exists():
            with path.open() as f:
                self.rows = [{k: float(v) if v else None for k, v in r.items()} for r in csv.DictReader(f)]

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_epoch = state.epoch or 0
        self.rows = [r for r in self.rows if r['epoch'] <= self.start_epoch + 1e-6]
        write_log(self.output / 'train_log.csv', self.rows)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.begin = time.perf_counter()
        self.first_step = state.global_step
        if torch.cuda.is_available() and not args.use_cpu:
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.stop_after and state.epoch >= self.start_epoch + self.stop_after - 1e-6:
            control.should_training_stop = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        elapsed = time.perf_counter() - self.begin
        records = [r for r in state.log_history if 'loss' in r and r.get('step', 0) > self.first_step]
        loss = float(np.mean([r['loss'] for r in records])) if records else None
        lr = records[-1].get('learning_rate') if records else None
        self.rows.append(dict(epoch=state.epoch, train_loss=loss, val_loss=metrics['eval_loss'], lr=lr,
                              epoch_seconds=elapsed, global_step=state.global_step,
                              cuda_peak_gb=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() and not args.use_cpu else 0.))
        write_log(self.output / 'train_log.csv', self.rows)
        mean = sum(r['epoch_seconds'] for r in self.rows) / len(self.rows)
        estimate = dict(measured_epoch_seconds=mean, planned_epochs=self.planned_epochs,
                        completed_epoch=state.epoch, remaining_train_seconds=max(0, self.planned_epochs-state.epoch)*mean,
                        projected_training_hours=mean*self.planned_epochs/3600,
                        note='Training+validation estimate; add zero-shot/final generation and checkpoint I/O. T4 target is 1–2 hours, not guaranteed.')
        (self.output / 'runtime_estimate.json').write_text(json.dumps(estimate, indent=2))
        print(f'\nMarian epoch {state.epoch:.2f}: {elapsed:.1f}s, val_loss={metrics["eval_loss"]:.4f}, '
              f'ETA={estimate["remaining_train_seconds"]/60:.1f} min', flush=True)


def run(config_path, stage='all', resume=None, checkpoint_dir=None, results_dir=None, stop_after_epochs=None, skip_test=False):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if checkpoint_dir:
        cfg['paths']['checkpoints'] = checkpoint_dir
    if results_dir:
        cfg['paths']['results'] = results_dir
    set_seed(cfg['seed'])
    tr = cfg['training']
    torch.set_num_threads(tr['cpu_threads'])
    device = torch.device('cuda' if torch.cuda.is_available() and tr['device'] != 'cpu' else 'cpu')
    if tr['device'] == 'cuda' and device.type != 'cuda':
        raise RuntimeError('Select a CUDA runtime; CPU validation uses a local tiny Marian fixture')
    root, output = Path(cfg['paths']['checkpoints']), Path(cfg['paths']['results'])
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = selected_splits(cfg)
    raw_hash = hashlib.sha256(Path(cfg['paths']['raw']).read_bytes()).hexdigest()
    identity = test_identity(rows['test'], raw_hash, cfg['seed'])
    # Paths/runtime knobs may differ after a reconnect, but freeze model/data/optimization and epoch schedule.
    contract = {k: cfg[k] for k in ('seed', 'model', 'data', 'training', 'evaluation')}
    contract = json.loads(json.dumps(contract))
    for key in ('device', 'cpu_threads', 'num_workers', 'pin_memory', 'progress'):
        contract['training'].pop(key, None)
    manifest_file = root / 'run_manifest.json'
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else None
    if manifest and (manifest['contract'] != contract or manifest['identity'] != identity):
        raise ValueError('Run config/data differs from manifest; use a new run directory')
    revision = manifest['resolved_revision'] if manifest else cfg['model']['revision']
    local = Path(cfg['model']['name']).is_dir()
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model']['name'], revision=revision).to(device)
    tokenizer_revision = getattr(model.config, '_commit_hash', None) or revision
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'], revision=tokenizer_revision, use_fast=False)
    if not manifest:
        resolved = getattr(model.config, '_commit_hash', None)
        if not local and not resolved:
            raise RuntimeError('Cannot identify the pretrained snapshot revision')
        if local:
            # Offline CPU fixture: explicitly not Helsinki pretrained model validation.
            resolved = 'local:' + hashlib.sha256((Path(cfg['model']['name'])/'model.safetensors').read_bytes()).hexdigest()
        manifest = dict(contract=contract, identity=identity, resolved_revision=resolved,
                        model_source=cfg['model']['name'], local_cpu_fixture=local)
        manifest_file.write_text(json.dumps(manifest, indent=2))
    zero_dir = output / 'zero_shot'
    zero_file = zero_dir / 'metrics.json'
    if not zero_file.exists():
        print('Evaluating base model BEFORE any fine-tuning', flush=True)
        evaluate(model, tokenizer, cfg, rows['test'], zero_dir, 'zero_shot', identity, manifest['resolved_revision'], 0.)
    else:
        old_zero = json.loads(zero_file.read_text())
        if old_zero['test_split_sha256'] != identity['test_split_sha256'] or old_zero['base_revision'] != manifest['resolved_revision']:
            raise ValueError('Zero-shot result does not match this run')
    if stage == 'zero-shot':
        return json.loads(zero_file.read_text())
    train_dir = root / 'trainer'
    train_dir.mkdir(exist_ok=True)
    last = get_last_checkpoint(str(train_dir))
    if resume == 'latest':
        if not last:
            raise FileNotFoundError('No Trainer checkpoint to resume')
        resume = last
    if resume and not Path(resume).is_dir():
        raise FileNotFoundError(resume)
    if not resume and (last or (root/'final').exists()):
        raise FileExistsError('Existing fine-tuning run; use --resume latest or a new run name')
    if resume and Path(resume).resolve().parent != train_dir.resolve():
        raise ValueError('Resume a checkpoint from this manifested run directory')
    final_dir = output / 'finetuned'
    final_dir.mkdir(exist_ok=True)
    epoch_log = EpochLog(final_dir, tr['epochs'], stop_after_epochs)
    train_dataset = ParallelDataset(rows['train'], tokenizer, cfg['data']['max_length'])
    val_dataset = ParallelDataset(rows['val'], tokenizer, cfg['data']['max_length'])
    args = Seq2SeqTrainingArguments(
        output_dir=str(train_dir), num_train_epochs=tr['epochs'],
        per_device_train_batch_size=tr['batch_size'], per_device_eval_batch_size=tr['eval_batch_size'],
        gradient_accumulation_steps=tr['gradient_accumulation_steps'],
        learning_rate=tr['learning_rate'], warmup_ratio=tr['warmup_ratio'], weight_decay=tr['weight_decay'],
        label_smoothing_factor=tr['label_smoothing'], max_grad_norm=tr['gradient_clip'],
        fp16=device.type=='cuda' and tr['mixed_precision'], use_cpu=device.type=='cpu',
        eval_strategy='epoch', save_strategy='epoch', logging_steps=tr['logging_steps'], logging_first_step=True,
        save_total_limit=2, save_safetensors=True, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, predict_with_generate=False,
        group_by_length=True, dataloader_num_workers=tr['num_workers'],
        dataloader_pin_memory=tr['pin_memory'] and device.type=='cuda',
        dataloader_persistent_workers=tr['num_workers']>0,
        report_to=[], push_to_hub=False, seed=cfg['seed'], data_seed=cfg['seed'],
        disable_tqdm=not tr['progress'], restore_callback_states_from_checkpoint=True)
    # Generation runs after loss-only validation; keep the pretrained cache available then.
    trainer = MarianTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=MarianCollator(tokenizer, model.config, 8 if device.type=='cuda' else None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr['patience']), epoch_log])
    print(json.dumps(dict(device=str(device), fp16=args.fp16, batch_size=tr['batch_size'],
                          effective_batch=tr['batch_size']*tr['gradient_accumulation_steps'],
                          resolved_revision=manifest['resolved_revision'], resume=resume)), flush=True)
    trainer.train(resume_from_checkpoint=resume)
    if skip_test:
        return json.loads((final_dir/'runtime_estimate.json').read_text())
    trainer.save_model(str(root/'final'))
    tokenizer.save_pretrained(root/'final')
    elapsed = sum(r['epoch_seconds'] for r in epoch_log.rows)
    return evaluate(trainer.model, tokenizer, cfg, rows['test'], final_dir, 'fine_tuned', identity,
                    manifest['resolved_revision'], elapsed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/helsinki_colab.yaml')
    p.add_argument('--stage', choices=['all','zero-shot','fine-tune'], default='all')
    p.add_argument('--resume', help='latest or a trusted checkpoint-N in this run')
    p.add_argument('--checkpoint-dir')
    p.add_argument('--results-dir')
    p.add_argument('--stop-after-epochs', type=int)
    p.add_argument('--skip-test', action='store_true')
    a = p.parse_args()
    metrics = run(a.config, a.stage, a.resume, a.checkpoint_dir, a.results_dir, a.stop_after_epochs, a.skip_test)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
