"""Audit UTF-8 TSV data and create deterministic source-grouped splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def group_key(text):
    # Case and punctuation variants must not cross split boundaries.
    return ''.join(c for c in normalize(text).casefold() if c.isalnum())


def read_raw(path):
    rows, widths, malformed = [], Counter(), 0
    with open(path, encoding='utf-8-sig') as f:
        for number, line in enumerate(f, 1):
            columns = line.rstrip('\r\n').split('\t')
            widths[len(columns)] += 1
            if len(columns) != 3:
                malformed += 1
                continue
            rows.append({'en': columns[0], 'ru': columns[1], 'attribution': columns[2], 'line': number})
    return rows, dict(widths), malformed


def clean(rows, max_chars=300, max_words=60):
    result, seen, reasons = [], set(), Counter()
    for row in rows:
        en, ru = normalize(row['en']), normalize(row['ru'])
        if not en or not ru:
            reasons['empty'] += 1
        elif not re.search('[A-Za-z]', en) or not re.search('[А-Яа-яЁё]', ru):
            reasons['missing_expected_script'] += 1
        elif any(unicodedata.category(c) in ('Cc', 'Cs') and not c.isspace() for c in en + ru):
            reasons['control_characters'] += 1
        elif max(len(en), len(ru)) > max_chars or max(len(en.split()), len(ru.split())) > max_words:
            reasons['length'] += 1
        elif (en, ru) in seen:
            reasons['duplicate_pair'] += 1
        else:
            seen.add((en, ru))
            result.append(dict(row, en=en, ru=ru))
    return result, dict(reasons)


def split_rows(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row['en'])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    cut1, cut2 = int(n * .9), int(n * .95)
    return {name: [r for k in subset for r in groups[k]] for name, subset in
            zip(('train', 'val', 'test'), (keys[:cut1], keys[cut1:cut2], keys[cut2:]))}


def distribution(values):
    import numpy as np
    return {k: float(v) for k, v in zip(('min', 'p50', 'p90', 'p95', 'p99', 'max'),
                                       np.percentile(values, [0, 50, 90, 95, 99, 100]))}


def prepare(raw, out, seed=42, report=None):
    raw, out = Path(raw), Path(out)
    rows, widths, malformed = read_raw(raw)
    if not rows:
        raise ValueError('No valid three-column TSV rows')
    cleaned, removed = clean(rows)
    splits = split_rows(cleaned, seed)
    if any(not v for v in splits.values()):
        raise ValueError('Not enough source groups for three nonempty splits')
    out.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    translations = defaultdict(set)
    for row in rows:
        translations[normalize(row['en'])].add(normalize(row['ru']))
    audit = {
        'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'seed': seed,
        'raw_lines': sum(widths.values()), 'column_counts': widths, 'malformed': malformed,
        'missing': {k: sum(not r[k].strip() for r in rows) for k in ('en', 'ru', 'attribution')},
        'noise_observations': {
            'non_nfc_rows': sum(any(unicodedata.normalize('NFC', r[k]) != r[k] for k in ('en', 'ru')) for r in rows),
            'whitespace_normalized_rows': sum(any(normalize(r[k]) != unicodedata.normalize('NFC', r[k]) for k in ('en', 'ru')) for r in rows),
            'identical_normalized_text': sum(normalize(r['en']).casefold() == normalize(r['ru']).casefold() for r in rows),
            'markup_like_rows': sum(bool(re.search(r'<[^>]+>|https?://', r['en'] + r['ru'])) for r in rows),
            'replacement_character_rows': sum('\ufffd' in r['en'] + r['ru'] for r in rows),
        },
        'raw_duplicate_pairs': len(rows) - len({(r['en'], r['ru']) for r in rows}),
        'sources_with_multiple_translations': sum(len(v) > 1 for v in translations.values()),
        'lengths': {lang: {unit: distribution([len(r[lang]) if unit == 'characters' else len(r[lang].split()) for r in rows])
                           for unit in ('characters', 'words')} for lang in ('en', 'ru')},
        'removed': removed, 'retained': len(cleaned),
        'split_rows': {k: len(v) for k, v in splits.items()},
        'split_groups': {k: len({group_key(r['en']) for r in v}) for k, v in splits.items()},
        'multitranslation_examples': [{'en': k, 'ru': sorted(v)[:5]} for k, v in translations.items() if len(v) > 1][:5],
    }
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text('# Data report\n\nGenerated from the checked-in `data/raw/rus.txt`; no download is required.\n\n'
            'Each row has English, Russian, and Tatoeba attribution, separated by tabs. Attribution is retained in processed rows. '
            'The supplied provenance is Kaggle / Tatoeba; the exact Kaggle URL is not supplied. Preserve the original attribution '
            'and review the dataset license before redistribution; rows identify CC-BY 2.0 (France).\n\n'
            '## Cleaning and leakage prevention\n\nNFC Unicode normalization and whitespace collapse; preserve case, punctuation, and Russian stress accents. '
            'Remove empty text, missing expected Latin/Cyrillic scripts, control characters, duplicate normalized pairs, '
            'and pairs over 300 characters or 60 whitespace words on either side. These are conservative heuristics, not semantic quality labels. '
            'No automatic language detector or manual quality certification is claimed.\n\n'
            'Group by casefolded English alphanumeric characters (punctuation/whitespace removed). Shuffle sorted groups with seed 42; '
            'allocate 90/5/5 percent of groups. All translations and simple punctuation/case variants stay together. '
            'Paraphrases can still overlap; sentence-level splitting does not guarantee topic isolation. Fit SentencePiece on train only.\n\n'
            '## Measured statistics\n\n```json\n' + json.dumps(audit, ensure_ascii=False, indent=2) + '\n```\n', encoding='utf-8')
    return splits


def load_rows(directory, split):
    with (Path(directory) / f'{split}.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw/rus.txt')
    p.add_argument('--out', default='data/processed/full')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--report', default='docs/data_report.md')
    a = p.parse_args()
    s = prepare(a.raw, a.out, a.seed, a.report)
    print({k: len(v) for k, v in s.items()})
