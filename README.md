# English → Russian machine translation

An end-to-end PyTorch portfolio project: audit a parallel corpus, fit a train-only SentencePiece tokenizer, and train an original Transformer implementation. This phase implements data exploration and a Transformer from scratch. **No full GPU training results are available yet.** Helsinki fine-tuning and a future notebook combining all stages are deferred until the scratch model is evaluated.

## Data

`data/raw/rus.txt` is checked in: 363,386 UTF-8 TSV lines containing English, Russian, and attribution. Supplied provenance: **Kaggle / Tatoeba**; an exact Kaggle listing was not provided. Original rows contain CC-BY 2.0 (France) attribution and Tatoeba contributor/sentence IDs. Preserve this column and consult the original license/attribution requirements before redistribution or publishing derived datasets. Do not interpret this repository as relicensing the corpus.

See [the measured data report](docs/data_report.md) for column counts, missing values, duplicate pairs, noise filters, length percentiles, multi-translation examples, source-group counts and dataset SHA-256. There is no dataset downloader or Kaggle API dependency.

## Fast Colab T4 workflow

The notebook now defaults to `configs/transformer_colab_fast.yaml`: 256 dimensions, three encoder/decoder layers, FFN 1024, 8k BPE, length cap 24, 12 epochs and patience 3. The **2–3 hour train+evaluation target is conditional, not measured on T4**. Train-only BPE pair-max p99 is 22 tokens; cap 24 truncates 0.49% of training pairs. Length buckets with a 16,384 padded source+target token budget reduce epoch-zero batches from 6,813 to 479 and padding from 47.9% to 10.8% (same vocabulary/cap comparison).

[Performance report](docs/performance_report.md) records CPU measurements, assumptions, batching rules and first-epoch calibration. Validation uses loss plus a fixed 128-source greedy subset. Final metrics report full-test greedy and a separate deterministic 1,024-source beam subset; they are never conflated.

```bash
python -m src.train --config configs/transformer_fast_cpu.yaml
# On T4: measure one real epoch first, without test evaluation.
python -u -m src.train --config configs/transformer_colab_fast.yaml --stop-after-epochs 1 --skip-test
python -u -m src.train --config configs/transformer_colab_fast.yaml --resume checkpoints/transformer_colab_fast/last.pt
```

Step progress/ETA, target tokens/sec, FP16 logits diagnostics, epoch time and peak GPU memory are logged. Notebook subprocess stdout/stderr stream live. Use a **new run name** for the new architecture; the old base checkpoints cannot resume into it. Latest train and validation loaders use spawned persistent workers and CUDA pinning. `runtime_estimate.json` projects the total from real epoch timing rather than CPU-to-GPU assumptions.

## Architecture

The original `transformer_base.yaml` configuration implements the post-norm encoder–decoder from *Attention Is All You Need*: six encoder and six decoder layers, `d_model=512`, eight attention heads, feed-forward width 2048, ReLU, sinusoidal positional encoding, residual connections and dropout. Scaled dot-product multi-head attention, encoder/decoder blocks, padding masks and causal masks are explicit PyTorch code; `nn.Transformer` is not used. In the original base model, English and Russian share a train-only 16,000-piece BPE vocabulary and input embedding. The output projection is independent.

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

For the retained original large baseline on a CUDA GPU (outside the fast budget):

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

A fresh run refuses an existing checkpoint directory. Choose `--checkpoint-dir` and `--results-dir` for a new experiment. Only load trusted checkpoints (`torch.load` uses pickle for optimizer/RNG state). Resume requires the same data hash, model, tokenizer, seed, sampling and optimization/batching settings. Epoch budget, device and loader/logging runtime settings may change. Legacy pipeline checkpoints remain evaluable but require a new training run after this batching/RNG change. Exact CPU epoch-boundary resume is tested; bitwise CUDA determinism and cross-device equivalence are not promised. Early-stopped runs remain stopped on resume. A disconnect mid-epoch replays that epoch. Keep Drive mounted and avoid concurrent writers; an interrupted Drive write should leave the preceding checkpoint intact, but Drive durability is not guaranteed by this code.

Training automatically evaluates **best validation-loss weights**, never the final epoch by default. `--skip-test` omits test decoding during a calibration run. Every completed run saves:

- `metrics.json`: test BLEU, chrF, signatures, training/evaluation time, data hash and full effective config.
- `train_log.csv`: epoch, train/validation loss, learning rate and epoch duration; restored on resume.
- `samples.txt`: 30 unique test sources, references and predictions (fewer only for a custom tiny test set).

Checkpoints include model/optimizer/scaler, Noam step, RNG state, best weights, early-stopping state, data hash, config and tokenizer bytes. Keep `last.pt` for resume; either checkpoint can evaluate best weights. Tokenizer files, raw data and predictions are not sent to an external service.

## Colab

Open [notebooks/train_transformer.ipynb](notebooks/train_transformer.ipynb) in Colab. All outputs are cleared. Select a T4 GPU and run cells in order: clone → install → mount Google Drive → inspect raw data → train/resume → inspect results. The notebook uses `feat/scratch-transformer` and fast-forwards a clean existing checkout on rerun; change `REPO_REF` to `main` after merge if desired. Use a unique fast run name; set `RESUME=True` when reconnecting to its saved `last.pt`. The notebook first measures one epoch, prints the total projection, then resumes the same run.

Fast-config planning target: **2–3 T4 hours including evaluation**, conditional on about 4.4k–6.8k target tokens/sec with the overhead assumptions in the performance report. The same architecture measured 1,337 target tokens/sec on two CPU threads; CPU speed alone cannot establish a T4 rate. Use the notebook's first-epoch measurement to replace the assumptions. Full GPU training, memory fit and Drive behavior remain unverified here. If OOM occurs, lower `max_tokens` in a copied config under a new experiment name. Batched decoding projects only the latest hidden position; prefixes still recompute without a KV cache.

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
- The retained base uses a 128-token cap (including BOS/EOS), preserving EOS on truncation. The fast config instead uses a train-measured p99-based cap of 24. The base cap predates the fast throughput audit. Character/word filtering precedes tokenization, so subword truncation can still happen. The original CPU cap is 48; the fast CPU smoke uses 24. Review length distributions before increasing caps.
- Retain the recognizable six-layer post-norm paper architecture as the original baseline; the fast preset uses three layers and 256 dimensions to fit a shorter budget; batch 48 and vocabulary 16k are T4-oriented choices, not measured optimal settings. Dropout, smoothing, Adam/Noam and clipping are configurable. Input embeddings are shared; output weights are not tied.
- Validation uses the same smoothed token loss as training, weighted by non-PAD target tokens. Best loss controls checkpoint selection and early stopping; test scores never control training or tuning.
- Multi-reference corpus evaluation groups exact English strings; missing references use sacreBLEU's `None` support. Samples are the deterministic first 30 shuffled unique test sources. Unfinished sequences at the decoding cap are scored as produced. Batched beam search keeps finished hypotheses and uses GNMT-style length penalty, no KV cache. The fast preset records full greedy and hash-selected beam-subset scores separately.
- Save complete epoch-boundary resume state and refuse accidental fresh-run checkpoint overwrite. Bundle tokenizer bytes so a Drive checkpoint is portable. Preprocessing is regenerated deterministically against a checked hash. Test training continuity with nonzero dropout.
- GPU learning belongs to the user. Automated development validation uses a tiny CPU subset only. No GPU quality claims, fabricated scores, fine-tuning stage or all-code combined notebook is included.
- This cloud task already has an isolated checkout; use it directly without creating additional Git worktrees.

## What to send after training

Send `metrics.json`, `train_log.csv`, `samples.txt`, `runtime_estimate.json`, the YAML you used, Git commit, GPU model, wall time, completed/best epoch, AMP dtype, tokens/sec, peak memory, token-budget changes, and any warnings/OOM/NaN messages. Keep checkpoints on Drive; share a checkpoint location only if further debugging needs it. We will inspect train/validation curves, tokenization/truncation and errors before the next stage.

## Decision after the first GPU run

These are provisional engineering gates, not promised benchmarks:

- Improve the scratch model first if validation loss is still falling at the epoch budget, a small training sample cannot be overfit, decoding is broken/repetitive, or train/validation divergence suggests a fixable optimization or regularization issue. Diagnose data/tokenization/masking before attributing failure to architecture.
- If loss converges cleanly, translations are coherent, and held-out BLEU ≥20 and chrF ≥45, retain the scratch model and focus on validation-guided error analysis and modest tuning.
- If an adequately trained, debugged scratch run plateaus below BLEU 15 or chrF 35, or available Colab compute is exhausted, consider Helsinki fine-tuning. Intermediate or conflicting scores call for manual sample review and one focused validation-driven improvement. Thresholds are dataset-specific heuristics; do not repeatedly tune against test scores. Reserve any new final holdout before extensive subsequent comparisons.

## Next stages

After GPU results are reviewed: decide whether to improve the scratch model or fine-tune `Helsinki-NLP/opus-mt-en-ru`; only then build the final notebook combining the project. Neither is implemented in this phase.
