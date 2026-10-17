"""Build Markdown/CSV comparisons, rejecting known test mismatches and labeling legacy runs."""
import argparse
import csv
import json
from pathlib import Path


def result_status(metrics):
    """Old Helsinki writers emitted metrics only after evaluation, but omitted status.

    Recovered metrics/samples are sufficient; never require or manufacture a training
    log. Explicit statuses (including failures) always take precedence. Completion
    describes execution, not model quality or uncontaminated pretraining.
    """
    status = metrics.get('training_status')
    if status and status != 'unreported':
        return status
    if (metrics.get('model_family') == 'helsinki_marian'
            and metrics.get('stage') in ('zero_shot', 'fine_tuned')
            and all(k in metrics for k in ('test_bleu', 'test_chrf', 'test_sources'))):
        return 'completed'
    return status or 'unreported'


def compare(paths, include_cpu=False, require_four=False):
    found = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            found.add(path)
        else:
            found.update(path.rglob('metrics.json'))
    metrics = [(p, json.loads(p.read_text())) for p in sorted(found)]
    metrics = [(p, m) for p, m in metrics if include_cpu or m.get('purpose') != 'cpu_smoke_only']
    if not metrics:
        raise ValueError('No result files (CPU smoke runs are excluded unless --include-cpu)')
    if require_four:
        roles = set()
        for _, m in metrics:
            cfg = m.get('config', {})
            family = m.get('model_family', '')
            experiment = m.get('experiment', cfg.get('experiment', ''))
            model = cfg.get('model', {})
            if family == 'scratch_transformer':
                if experiment.startswith('scratch_fast') or (model.get('d_model') == 256 and model.get('layers') == 3):
                    roles.add('fast')
                if experiment.startswith('scratch_large') or (model.get('d_model') == 512 and model.get('layers') == 6):
                    roles.add('large')
            elif family == 'helsinki_marian' or (include_cpu and family == 'marian_cpu_fixture'):
                if m.get('stage') in ('zero_shot', 'fine_tuned'):
                    roles.add(m['stage'])
        missing = {'fast','large','zero_shot','fine_tuned'} - roles
        if missing:
            raise ValueError(f'Missing required comparison runs: {sorted(missing)}')
        if not include_cpu:
            for path, m in metrics:
                if m.get('schema_version') != 2 or not m.get('test_split_sha256'):
                    raise ValueError(f'Required comparison needs schema v2 exact test identity: {path}')
                if (m.get('test_decoding') != 'beam' or m.get('test_beam_size') != 4
                        or 'test_greedy_bleu' not in m or 'test_greedy_chrf' not in m):
                    raise ValueError(f'Required comparison needs full-test beam4 AND greedy: {path}')
    hashes = {m['test_split_sha256'] for _, m in metrics if m.get('test_split_sha256')}
    if len(hashes) > 1:
        raise ValueError('Different held-out reference sets: compare these in separate tables')
    missing_hash_counts = {m.get('config',{}).get('data',{}).get('max_test') for _,m in metrics}
    if len(missing_hash_counts)>1 and any(not m.get('test_split_sha256') for _,m in metrics):
        raise ValueError('Legacy runs with different test sampling cannot be verified')
    raw = {m['data_sha256'] for _, m in metrics if m.get('data_sha256')}
    seeds = {m.get('split_seed', m.get('config', {}).get('seed')) for _, m in metrics}
    seeds.discard(None)
    if len(raw)>1 or len(seeds)>1:
        raise ValueError('Different raw dataset or split seed')
    for field in ('bleu_signature', 'chrf_signature'):
        signatures = {m[field] for _, m in metrics if m.get(field)}
        if len(signatures)>1:
            raise ValueError(f'Different metric signatures: {field}')
    counts = {m['test_sources'] for _, m in metrics}
    if len(counts)>1:
        raise ValueError('Different test source counts')
    rows = []
    for path, m in metrics:
        identity = 'verified' if m.get('test_split_sha256') else 'legacy/unverified'
        policy = m.get('test_decoding', 'unknown')
        ngram = m.get('no_repeat_ngram_size', 0)
        if ngram:
            policy += f'/no-repeat-{ngram}'
        rows.append(dict(run=str(path.parent), model=m.get('experiment') or m.get('model_family', 'legacy'), status=result_status(m), stage=m.get('stage','legacy'), policy=policy,
                         beam=m.get('test_beam_size', m.get('config',{}).get('evaluation',{}).get('beam_size','?')),
                         bleu=m['test_bleu'], chrf=m['test_chrf'], train_seconds=m['training_seconds'],
                         eval_seconds=m.get('evaluation_seconds', 0.), sources=m['test_sources'], identity=identity))
        if m.get('test_decoding') == 'beam' and 'test_greedy_bleu' in m:
            rows.append(dict(rows[-1], policy=f'greedy/no-repeat-{ngram}' if ngram else 'greedy', beam=1, bleu=m['test_greedy_bleu'], chrf=m['test_greedy_chrf']))
        if 'repetition_ablation' in m:
            a = m['repetition_ablation']
            rows.append(dict(rows[-1], policy=f'beam/no-repeat-{a["no_repeat_ngram_size"]}', beam=a['beam_size'],
                             bleu=a['bleu'], chrf=a['chrf']))
        # Subset scores are deliberately excluded from the full-test comparison.
    return rows


def write_comparison(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| Run | Model | Status | Stage | Decode | Beam | BLEU | chrF | Train s | Eval s | Sources | Identity |',
             '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f'| {r["run"].replace("|", "/")} | {r["model"]} | {r["status"]} | {r["stage"]} | {r["policy"]} | {r["beam"]} | '
                     f'{r["bleu"]:.2f} | {r["chrf"]:.2f} | {r["train_seconds"]:.1f} | '
                     f'{r["eval_seconds"]:.1f} | {r["sources"]} | {r["identity"]} |')
    lines += ['', 'Times are per-run totals repeated per policy; do not sum rows. Full-test scores only. Legacy files lack exact reference-set verification. '
              'Legacy Helsinki evaluation artifacts imply completed execution even if the recovered training log is absent. Helsinki may have Tatoeba pretraining contamination. Different tokenizers/caps/decoding policies are not controlled ablations.']
    text = '\n'.join(lines)+'\n'
    output.with_suffix('.md').write_text(text)
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', help='metrics.json files or result directories (recursive)')
    p.add_argument('--output', default='results/comparison')
    p.add_argument('--include-cpu', action='store_true')
    p.add_argument('--require-four-models', action='store_true', help='Require scratch fast, large, Helsinki zero-shot and fine-tuned')
    a = p.parse_args()
    print(write_comparison(compare(a.paths, a.include_cpu, a.require_four_models), a.output))
