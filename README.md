# English → Russian machine translation

An end-to-end PyTorch portfolio project: audit a parallel corpus, fit a train-only SentencePiece tokenizer, and train an original Transformer implementation. This phase implements data exploration and a Transformer from scratch. **No full GPU training results are available yet.** Helsinki fine-tuning and a future notebook combining all stages are deferred until the scratch model is evaluated.

## Data

`data/raw/rus.txt` is checked in: 363,386 UTF-8 TSV lines containing English, Russian, and attribution. Supplied provenance: **Kaggle / Tatoeba**; an exact Kaggle listing was not provided. Original rows contain CC-BY 2.0 (France) attribution and Tatoeba contributor/sentence IDs. Preserve this column and consult the original license/attribution requirements before redistribution or publishing derived datasets. Do not interpret this repository as relicensing the corpus.

See [the measured data report](docs/data_report.md) for column counts, missing values, duplicate pairs, noise filters, length percentiles, multi-translation examples, source-group counts and dataset SHA-256. There is no dataset downloader or Kaggle API dependency.

## Architecture

The base configuration implements the post-norm encoder–decoder from *Attention Is All You Need*: six encoder and six decoder layers, `d_model=512`, eight attention heads, feed-forward width 2048, ReLU, sinusoidal positional encoding, residual connections and dropout. Scaled dot-product multi-head attention, encoder/decoder blocks, padding masks and causal masks are explicit PyTorch code; `nn.Transformer` is not used. English and Russian share a train-only 16,000-piece BPE vocabulary and input embedding. The output projection is independent.

Training uses teacher forcing, label-smoothed cross entropy (PAD ignored), Adam (β₁=.9, β₂=.98, ε=1e-9), Noam warmup, gradient clipping, CUDA FP16 AMP and validation-loss early stopping. Greedy and length-normalized beam search are implemented. sacreBLEU (13a tokenization) and chrF use every available Russian reference for each exact English source. Metric signatures are recorded.

## Local usage

Run commands from the repository root with Python 3.12. CPU needs no GPU runtime:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only installation avoids CUDA wheel downloads; omit this line on GPU.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.data
python -m pytest -q
python -m src.train --config configs/transformer_cpu.yaml
```

The CPU configuration uses 96 training pairs, 32 validation pairs and 40 test pairs selected **after** grouped splitting. Its two tiny epochs validate plumbing, not translation quality. Outputs go to `results/transformer_cpu/`; do not present these scores as a trained model benchmark.

On a CUDA GPU:

```bash
python -m src.train --config configs/transformer_base.yaml
# Resume at the last completed epoch; epochs may be increased in YAML.
python -m src.train --config configs/transformer_base.yaml \
  --resume checkpoints/transformer_base/last.pt
python -m src.evaluate --checkpoint checkpoints/transformer_base/last.pt
python -m src.translate --checkpoint checkpoints/transformer_base/last.pt \
  --text "Hello, how are you?" --beam-size 4
# Greedy comparison in a separate directory:
python -m src.evaluate --checkpoint checkpoints/transformer_base/last.pt \
  --beam-size 1 --results-dir results/greedy
```

A fresh run refuses an existing checkpoint directory. Choose `--checkpoint-dir` and `--results-dir` for a new experiment. Only load trusted checkpoints (`torch.load` uses pickle for optimizer/RNG state). Resume requires the same data hash, model, tokenizer, seed, sampling and training settings (except total epochs, device and CPU thread count). Exact CPU epoch-boundary resume is tested; bitwise CUDA determinism and cross-device equivalence are not promised. Early-stopped runs remain stopped on resume. A disconnect mid-epoch replays that epoch. Keep Drive mounted and avoid concurrent writers; an interrupted Drive write should leave the preceding checkpoint intact, but Drive durability is not guaranteed by this code.

Training automatically evaluates **best validation-loss weights**, never the final epoch by default. Every completed run saves:

- `metrics.json`: test BLEU, chrF, signatures, training/evaluation time, data hash and full effective config.
- `train_log.csv`: epoch, train/validation loss, learning rate and epoch duration; restored on resume.
- `samples.txt`: 30 unique test sources, references and predictions (fewer only for a custom tiny test set).

Checkpoints include model/optimizer/scaler, Noam step, RNG state, best weights, early-stopping state, data hash, config and tokenizer bytes. Keep `last.pt` for resume; either checkpoint can evaluate best weights. Tokenizer files, raw data and predictions are not sent to an external service.

## Colab

Open [notebooks/train_transformer.ipynb](notebooks/train_transformer.ipynb) in Colab. All outputs are cleared. Select a T4 GPU and run cells in order: clone → install → mount Google Drive → inspect raw data → train/resume → inspect results. The notebook uses this PR branch until merge; change `REPO_REF` to `main` after merge. Set a unique run name; set `RESUME=True` when reconnecting to its saved `last.pt`.

Planning estimate only: 12–48 T4 GPU-hours for 30 epochs plus 0.5–3 hours for serial beam evaluation; this has **not** been benchmarked. Estimate the real budget from the first epoch; Colab session limits may require repeated resumes. Full GPU training, T4 memory fit, AMP and the Colab/Drive integration require your run. If memory is insufficient, lower batch size in a copied config under a new experiment name and record that change. Evaluation is serial and favors simplicity over throughput.

## Results

| Run | Test sacreBLEU ↑ | Test chrF ↑ | Training time | Status |
| --- | --- | --- | --- | --- |
| Scratch Transformer, T4 | TBD | TBD | TBD | Awaiting user training |
| Helsinki-NLP/opus-mt-en-ru fine-tuning | TBD | TBD | TBD | Future stage |

See [CPU validation evidence](docs/cpu_validation.md) for development checks, separate from model quality results.

## Design decisions

- Use the checked-in data as the source of truth; no API/download step or invented data is substituted. Keep original attribution in processed rows.
- Normalize Unicode to NFC and collapse whitespace. Preserve case, punctuation and Russian combining stress marks; apply conservative expected-script, control-character, 300-character and 60-word filters. Remove exact normalized pair duplicates. All rule counts are in the report.
- Split **source groups**, not pairs: English casefolded alphanumerics define a group. Seed 42 shuffles sorted groups into 90/5/5 proportions by group count; row proportions differ. All alternative Russian translations remain together. Similar paraphrases are not guaranteed isolated.
- Fit a joint BPE tokenizer only on the training split. BOS/EOS/PAD/UNK have explicit IDs. Vocabulary uses `hard_vocab_limit=False` so a tiny corpus may create fewer pieces. No validation/test vocabulary fitting.
- Use a 128-token cap (including BOS/EOS), preserving EOS on truncation. This is a compute compromise for T4. Character/word filtering precedes tokenization, so subword truncation can still happen. The CPU cap is 48. Review length distributions before increasing caps.
- Keep the recognizable six-layer post-norm paper architecture; batch 48 and vocabulary 16k are T4-oriented choices, not measured optimal settings. Dropout, smoothing, Adam/Noam and clipping are configurable. Input embeddings are shared; output weights are not tied.
- Validation uses the same smoothed token loss as training, weighted by non-PAD target tokens. Best loss controls checkpoint selection and early stopping; test scores never control training or tuning.
- Multi-reference corpus evaluation groups exact English strings; missing references use sacreBLEU's `None` support. Samples are the deterministic first 30 shuffled unique test sources. Unfinished sequences at the decoding cap are scored as produced. Beam search keeps finished hypotheses and uses GNMT-style length penalty, no KV cache.
- Save complete epoch-boundary resume state and refuse accidental fresh-run checkpoint overwrite. Bundle tokenizer bytes so a Drive checkpoint is portable. Preprocessing is regenerated deterministically against a checked hash. Test training continuity with nonzero dropout.
- GPU learning belongs to the user. Automated development validation uses a tiny CPU subset only. No GPU quality claims, fabricated scores, fine-tuning stage or all-code combined notebook is included.
- This cloud task already has an isolated checkout; use it directly without creating additional Git worktrees.

## What to send after training

Send `metrics.json`, `train_log.csv`, `samples.txt`, the YAML you used, Git commit, GPU model, wall time, completed/best epoch, batch-size changes, and any warnings/OOM/NaN messages. Keep checkpoints on Drive; share a checkpoint location only if further debugging needs it. We will inspect train/validation curves, tokenization/truncation and errors before the next stage.

## Decision after the first GPU run

These are provisional engineering gates, not promised benchmarks:

- Improve the scratch model first if validation loss is still falling at the epoch budget, a small training sample cannot be overfit, decoding is broken/repetitive, or train/validation divergence suggests a fixable optimization or regularization issue. Diagnose data/tokenization/masking before attributing failure to architecture.
- If loss converges cleanly, translations are coherent, and held-out BLEU ≥20 and chrF ≥45, retain the scratch model and focus on validation-guided error analysis and modest tuning.
- If an adequately trained, debugged scratch run plateaus below BLEU 15 or chrF 35, or available Colab compute is exhausted, consider Helsinki fine-tuning. Intermediate or conflicting scores call for manual sample review and one focused validation-driven improvement. Thresholds are dataset-specific heuristics; do not repeatedly tune against test scores. Reserve any new final holdout before extensive subsequent comparisons.

## Next stages

After GPU results are reviewed: decide whether to improve the scratch model or fine-tune `Helsinki-NLP/opus-mt-en-ru`; only then build the final notebook combining the project. Neither is implemented in this phase.
