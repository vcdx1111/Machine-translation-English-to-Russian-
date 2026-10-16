> Historical initial implementation evidence. Current stable scratch/Helsinki comparison and checks are in [stability_diagnosis.md](stability_diagnosis.md).

# Add data audit and an English→Russian Transformer from scratch

## Implemented

- Inspect the real checked-in 363,386-line English/Russian/attribution TSV; publish measured audit statistics, normalization/noise rules and multi-translation examples in `docs/data_report.md`.
- Deterministic seed-42 source-group train/val/test splits: 327,013 / 18,248 / 18,124 pairs. Keep case/punctuation variants and alternative translations together; fit SentencePiece BPE only on train.
- Explicit PyTorch scaled dot-product multi-head attention, sinusoidal positions, post-norm encoder/decoder, causal/padding masks; no `nn.Transformer`.
- Label smoothing, Adam/Noam warmup, clipping, CUDA AMP, early stopping, atomic checkpoint replacement and complete epoch-boundary resume (optimizer/scaler/RNG/tokenizer/data hash/best weights).
- Greedy and length-normalized beam search; multi-reference sacreBLEU and chrF with signatures. Automatic `metrics.json`, `train_log.csv`, and 30-example `samples.txt`.
- Base T4 and tiny CPU YAML configs, pinned requirements, English README and design decisions, output-free Colab training notebook with Google Drive checkpoint/resume.
- No dataset downloader, Helsinki fine-tuning, or final combined notebook.

## Validation and actual CPU logs

```text
python -m pytest -q
..........                                                               [100%]
10 passed in 5.00s

python -m src.data
train=327013, val=18248, test=18124

python -m src.train --config configs/transformer_cpu.yaml
```

Actual `train_log.csv` from the CPU smoke run:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds
1,5.071299634585723,4.803919864578481,0.03314563036811942,0.17059502099982637
2,4.782265285221792,4.614546905997341,0.036084391824351615,0.21700006600008237
```

CPU smoke only: 96 train / 32 val / 40 test pairs; BLEU **0.0**, chrF **2.680311890838207**, 0.388 seconds of epoch computation, 1.907 seconds of evaluation. These values do not represent translation model quality. Outputs were verified, including exactly 30 samples. Standalone checkpoint evaluation reproduces the same scores; greedy CLI completes. All full-data split group intersections are empty. `pip check`, notebook JSON/schema/cleared-output validation, and `git diff --check` pass. See `docs/cpu_validation.md` for reproducible commands and full evidence.

Tests cover masks, shape/causality/padding invariance, tokenization, data cleaning/group isolation, a training step, beam-vs-greedy behavior, exact CPU resume with dropout, multi-reference metrics and notebook validity.

**Not verified:** full GPU learning, translation quality, T4 memory/throughput, CUDA mixed precision behavior, and actual Colab/Drive integration. GPU learning is intentionally left to the user. Results table remains TBD.

## Colab run order and expected time

1. Open `notebooks/train_transformer.ipynb` and select a T4 GPU.
2. Clone this implementation branch (change the notebook ref to `main` after merge), install pinned packages, confirm CUDA.
3. Mount Drive, choose a unique run name, inspect the checked-in raw TSV and audit.
4. Run training with `configs/transformer_base.yaml`. Automatic best-validation checkpoint evaluation writes results to Drive.
5. On reconnect, rerun setup cells with the same run name and `RESUME=True`; `last.pt` resumes at a completed epoch. An interrupted epoch is replayed. If only evaluation was interrupted, use the optional standalone evaluation cell.
6. Inspect `metrics.json`, `train_log.csv`, and `samples.txt` in the final cell.

Planning estimate, **not benchmarked**: 12–48 T4 GPU-hours for 30 epochs, plus 0.5–3 hours for serial beam evaluation. Measure the first epoch and extrapolate; early stopping and sentence lengths change the budget. Colab session limits likely require resumes. If OOM occurs, use a copied YAML with smaller batch size and a new run name, and record the changed config.

## Send back after training

- `metrics.json`, `train_log.csv`, `samples.txt`.
- Exact YAML and Git commit; GPU model, wall time, completed/best epoch.
- Batch-size/config changes, warnings, OOM/NaN/errors, and interruption/resume details.
- Keep trusted `last.pt`/`best.pt` and tokenizer on Drive; checkpoint sharing is only needed if debugging requires it.

## Decision gate after the run

Use validation curves and manual sample review before deciding. Fix optimization/data/tokenizer/decoding problems or train longer when validation loss is still improving. Inability to overfit a small subset or pathological repetition warrants debugging, not immediately switching models.

Provisional dataset-specific gates: BLEU ≥20 **and** chrF ≥45 with coherent samples favors retaining and improving the scratch model. A clean, adequately trained plateau below BLEU 15 **or** chrF 35, or exhausted Colab compute, favors discussing `Helsinki-NLP/opus-mt-en-ru` fine-tuning. Intermediate/conflicting metrics call for one focused validation-guided iteration and error analysis. Avoid repeated test-driven tuning; reserve a fresh final holdout before extensive subsequent comparisons.

Helsinki fine-tuning and the final combined notebook will start only after the user's results are reviewed.
