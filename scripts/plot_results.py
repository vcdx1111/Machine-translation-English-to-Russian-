"""Export publication figures from real logs and metrics, or reported aggregate scores.

The aggregate JSON is explicitly labelled; epoch curves always require a real CSV.
"""
import argparse
import csv
import json
from pathlib import Path


def plot_results(summary, loss_log, output='docs/figures', results_root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(Path(summary).read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = summary['runs']
    names = [r['name'].replace('Transformer ', 'Scratch ').replace('Helsinki ', 'Helsinki\n') for r in runs]
    bars = []
    for r in runs:
        if results_root:
            path = Path(results_root) / r['path'] / 'metrics.json'
            m = json.loads(path.read_text(encoding='utf-8'))
            if (m['test_sources'] != summary['test_sources'] or
                    m['test_split_sha256'] != summary.get('test_split_sha256', m['test_split_sha256']) or
                    not m['test_split_sha256'].startswith(summary['test_split_sha256_prefix'])):
                raise ValueError(f'Different held-out set: {path}')
            if m.get('test_decoding') != 'beam' or m.get('test_beam_size') != 4:
                raise ValueError(f'Expected full-test beam4: {path}')
            bars.append((m['test_greedy_bleu'], m['test_bleu'], m['test_greedy_chrf'], m['test_chrf']))
        else:
            bars.append((r['greedy_bleu'], r['beam_bleu'], r['greedy_chrf'], r['beam_chrf']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(runs)); width = .36
    for ax, title, offset in zip(axes, ('sacreBLEU (13a)', 'chrF2'), (0, 2)):
        for k, label in enumerate(('Greedy', 'Beam 4')):
            values = [b[offset+k] for b in bars]
            rectangles = ax.bar(x + (k-.5)*width, values, width, label=label)
            ax.bar_label(rectangles, fmt='%.2f', fontsize=8)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylim(0, 85)
        ax.set_ylabel(title)
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'EN→RU · {summary["test_sources"]:,} held-out sources · multiple references')
    fig.savefig(output/'model_scores.png', dpi=180)
    plt.close(fig)
    if loss_log:
        with Path(loss_log).open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError('Empty loss CSV; no curve can be reconstructed')
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for key, label in [('train_loss','Train'), ('val_loss','Validation')]:
            ax.plot([float(r['epoch']) for r in rows], [float(r[key]) for r in rows], label=label)
        ax.set(xlabel='Epoch', ylabel='Label-smoothed cross-entropy', title='Transformer large v4: validation loss converged')
        ax.legend(); ax.spines[['top','right']].set_visible(False)
        fig.savefig(output/'large_v4_loss.png', dpi=180)
        plt.close(fig)
    provenance = {'scores_source':'Drive metrics.json' if results_root else summary['provenance'],
                  'loss_source':str(loss_log) if loss_log else None,
                  'note':'No missing epoch values have been interpolated or fabricated.'}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', default='docs/final_results.json')
    p.add_argument('--loss-log', help='Real scratch_large_stable_v4/results/train_log.csv')
    p.add_argument('--results-root', help='Drive en_ru_translation directory; omit to plot reported aggregate scores')
    p.add_argument('--output', default='docs/figures')
    a=p.parse_args()
    print(plot_results(a.summary,a.loss_log,a.output,a.results_root))
