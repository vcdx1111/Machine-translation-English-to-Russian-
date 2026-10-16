"""Shared result artifacts and exact held-out reference identity across model families."""
import hashlib
import json
from pathlib import Path


def test_identity(rows, data_hash, seed):
    # Invariant to batching/order, sensitive to every source/reference pair.
    pairs = sorted({(r['en'], r['ru']) for r in rows})
    payload = json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode()
    return dict(schema_version=2, data_sha256=data_hash, split_seed=seed,
                test_split_sha256=hashlib.sha256(payload).hexdigest(),
                test_pairs=len(rows), test_sources=len({s for s, _ in pairs}),
                metric_policy='multi-reference sacreBLEU 13a + chrF2; exact English source grouping')


def write_artifacts(output, metrics, sources, references, predictions, samples_name='samples.txt'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / samples_name).open('w', encoding='utf-8') as f:
        for source, hyp in list(zip(sources, predictions))[:30]:
            f.write(f'EN: {source}\nREF: {" || ".join(references[source])}\nMODEL: {hyp}\n\n')
