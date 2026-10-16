> Historical initial implementation evidence. The current fast pipeline, tests and time estimates are documented in [performance_report.md](performance_report.md).

# CPU validation evidence

This is a functional smoke run, **not a trained translation benchmark**. No GPU training was performed. The raw dataset remains unchanged.

Environment: Python 3.12.14, PyTorch 2.6.0+cpu, no CUDA. Packages installed from pinned `requirements.txt` (CPU PyTorch wheel from the official index).

## Commands and observed results

```bash
python -m pytest -q
..........                                                               [100%]
10 passed in 5.00s
```

Tests cover causal/padding masks, attention exclusion, tensor shapes, future-token and source-padding invariance, TSV cleaning, source-group split isolation, SentencePiece roundtrip/special IDs, a training update, greedy versus beam search and EOS, exact epoch-boundary resume with dropout, multi-reference BLEU/chrF including missing references, and notebook JSON/schema/cleared outputs.

```bash
python -m src.data
# train=327013, val=18248, test=18124
python -m src.train --config configs/transformer_cpu.yaml
```

Actual `train_log.csv`:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds
1,5.071299634585723,4.803919864578481,0.03314563036811942,0.17059502099982637
2,4.782265285221792,4.614546905997341,0.036084391824351615,0.21700006600008237
```

Observed result summary:

```json
{
  "purpose": "cpu_smoke_only",
  "test_bleu": 0.0,
  "test_chrf": 2.680311890838207,
  "test_sources": 40,
  "test_pairs": 40,
  "training_seconds": 0.38759508699990874,
  "evaluation_seconds": 1.9070979220000481,
  "data_sha256": "d5105634aef975c9ddc9bfa0862cc8f882cd29fa7a97bde2a5e3af18e5ad0504"
}
```

Verified all three outputs exist in `results/transformer_cpu/`: valid `metrics.json`, two-epoch `train_log.csv`, and `samples.txt` containing exactly 30 source/reference/output blocks. These local generated files are ignored. CPU scores above only demonstrate execution and must not enter the README GPU quality table.

```bash
python -m src.evaluate --checkpoint checkpoints/transformer_cpu/last.pt \
  --results-dir results/transformer_cpu_recheck
python -m src.translate --checkpoint checkpoints/transformer_cpu/last.pt \
  --text 'Hello!' --beam-size 1
```

Standalone evaluation reproduced identical BLEU/chrF from bundled tokenizer and best checkpoint weights. Greedy translation CLI completed; its tiny-run output has no quality significance. A direct check on **all** processed full-data rows confirmed no source-group intersection between any split pair. The notebook validates against nbformat v4, has no execution counts or outputs, and every code cell has preceding Markdown.

## Not validated here

Full-corpus GPU optimization, T4 throughput/memory fit, CUDA AMP overflow behavior, translation quality, and actual Colab/Google Drive execution remain for the user's GPU run. The 12–48 GPU-hour training estimate and 0.5–3 hour beam evaluation estimate are planning ranges, not measured results. The checkpoint replay test is CPU-only. No Helsinki fine-tuning or combined notebook is present.
