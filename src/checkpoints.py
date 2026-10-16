"""Retain one compact best checkpoint only AFTER completed evaluation; cannot resume it."""
import argparse
import json
from pathlib import Path
import torch
from .train import save_checkpoint


def prune_to_best(checkpoints, results):
    checkpoints, results = Path(checkpoints), Path(results)
    status = json.loads((results/'run_status.json').read_text())
    metrics = json.loads((results/'metrics.json').read_text())
    if status['status'] not in ('completed', 'early_stopped', 'pruned') or not status.get('evaluation_complete'):
        raise ValueError('Keep resume state until training AND evaluation complete; diverged/running/calibrated runs are not pruned')
    path = checkpoints/'best.pt'
    best = torch.load(path, map_location='cpu', weights_only=False)
    if not best.get('run_id') or best['run_id'] != metrics.get('run_id') or best['run_id'] != status.get('run_id'):
        raise ValueError('Best checkpoint/result/status run IDs do not match')
    if best['data_sha256'] != metrics['data_sha256'] or best['config']['model'] != metrics['config']['model']:
        raise ValueError('Checkpoint and evaluated results do not match')
    if best['config']['seed'] != metrics['config']['seed']:
        raise ValueError('Checkpoint/result seeds do not match')
    last = checkpoints/'last.pt'
    if last.exists():
        candidate = torch.load(last, map_location='cpu', weights_only=False)
        if (candidate.get('run_id') != best['run_id'] or candidate['data_sha256'] != best['data_sha256']):
            raise ValueError('last.pt belongs to a different experiment')
    before = sum(p.stat().st_size for p in (path,last) if p.exists())
    weights = dict(best['best_model'])
    if best['config']['model'].get('tie_embeddings'):
        if not torch.equal(weights['embedding.weight'], weights['output.weight']):
            raise ValueError('Tied checkpoint matrices differ')
        weights['output.weight'] = weights['embedding.weight']
    training_metadata = {k: metrics[k] for k in ('run_id','experiment','model_variant','training_status','stop_reason',
                         'best_epoch','completed_epochs','training_seconds','attempted_steps','optimizer_steps') if k in metrics}
    compact = dict(training_metadata=training_metadata, run_id=best['run_id'], artifact_type='inference_best', resume_supported=False,
                   pipeline_version=best.get('pipeline_version'), config=metrics['config'],
                   best_model=weights, tokenizer_model=best['tokenizer_model'],
                   data_sha256=best['data_sha256'], best_epoch=metrics['best_epoch'],
                   training_seconds=metrics['training_seconds'], history=best['history'],
                   completed_epochs=metrics['completed_epochs'])
    # Verify loadability before deleting resume weights. Only these known artifacts are removed.
    save_checkpoint(path,compact)
    verified = torch.load(path,map_location='cpu',weights_only=False)
    if verified['artifact_type'] != 'inference_best' or not verified['best_model']:
        raise RuntimeError('Compact best checkpoint verification failed; last.pt retained')
    for name in ('last.pt','last.tmp','best.tmp'):
        (checkpoints/name).unlink(missing_ok=True)
    report = dict(run_id=best['run_id'],status='pruned',evaluation_complete=True,resume_supported=False,
                  best_epoch=metrics['best_epoch'],bytes_before=before,bytes_after=path.stat().st_size)
    (results/'run_status.json').write_text(json.dumps(report,indent=2))
    (results/'checkpoint_cleanup.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--results-dir',required=True)
    a=p.parse_args()
    print(json.dumps(prune_to_best(a.checkpoint_dir,a.results_dir),indent=2))
