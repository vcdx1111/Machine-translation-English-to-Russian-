"""Join exact model outputs by English source; missing outputs are never invented."""
import argparse
import json
from pathlib import Path

SOURCES = [
    'The plans were discarded.',
    'Sometimes cows are killed by coyotes.',
    'They have lost 10 games in a row since their winning streak ended.',
    'She quickly put the money out of sight.',
    'Tom is clearly lost.',
]


def read_samples(path):
    # Standard EN/REF/MODEL format; files may contain only the first 30 test sources.
    samples = {}
    if not Path(path).exists():
        return samples
    source = None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('EN: '):
            source=line[4:]
        elif line.startswith('MODEL: ') and source is not None:
            samples[source]=line[7:]
    return samples


def example_table(root, summary='docs/final_results.json', sources=None, include_common=True):
    runs=json.loads(Path(summary).read_text())['runs']
    maps=[read_samples(Path(root)/r['path']/'samples.txt') for r in runs]
    chosen=list(SOURCES if sources is None else sources)
    if include_common and maps:
        common=set.intersection(*(set(m) for m in maps))
        chosen += sorted(common-set(chosen))[:10]
    def escaped(x):
        return x.replace('|', '\\|').replace('\n',' ')
    lines=['| English | '+' | '.join(r['name'] for r in runs)+' |', '| '+' | '.join(['---']*(len(runs)+1))+' |']
    for source in chosen:
        lines.append('| '+' | '.join(escaped(t) for t in [source]+[m.get(source,'Exact output unavailable in samples.txt') for m in maps])+' |')
    lines += ['', 'Only verbatim saved outputs are shown. The standard samples.txt stores 30 sources; requested error cases may be absent. '
              'Missing translations require an existing prediction artifact or inference with saved best weights, not invented text.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',required=True)
    p.add_argument('--summary',default='docs/final_results.json')
    p.add_argument('--output',default='docs/translation_examples.md')
    a=p.parse_args(); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(example_table(a.results_root,a.summary),encoding='utf-8')
    print(output)
