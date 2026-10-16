# English → Russian machine translation

A PyTorch portfolio project comparing a **Transformer implemented from scratch**, a larger scratch model, and **Helsinki-NLP/opus-mt-en-ru zero-shot / fine-tuning**. Attention, sinusoidal positions, encoder/decoder blocks and masks are explicit PyTorch code; `nn.Transformer` is not used. Helsinki uses its pretrained Marian architecture and `Seq2SeqTrainer`.

GPU learning runs in the user's Colab. Development checks here use CPU subsets and a local random tiny Marian fixture. The final notebook combining every stage is still deferred; the two current notebooks are separate runnable workflows.

## Data and evaluation

`data/raw/rus.txt` is tracked: 363,386 UTF-8 TSV rows with **English, Russian, attribution**. Supplied provenance is Kaggle / Tatoeba; the exact Kaggle listing was not provided. Original rows contain CC-BY 2.0 (France) and contributor/sentence attribution. Preserve those credits and review the original license before redistributing derived data. No Kaggle API or data downloader is included.

[Data report](docs/data_report.md): cleaning retains 363,385 pairs; seed-42 source-group splits contain **327,013 train / 18,248 validation / 18,124 test pairs**. Casefolded English alphanumerics define a group, keeping punctuation/case variants and alternative Russian translations together. All models call the same split/sampling function. Tokenizers, token caps and models never change the held-out references. Paraphrase-level overlap is not ruled out.

Evaluation shares **multi-reference sacreBLEU (13a) and chrF2**, grouping exact English strings and passing all available Russian references. Metrics include signatures, raw SHA-256, split seed and an exact test source/reference-set hash. Large scratch and Helsinki report full-test beam and greedy; the old fast run's 1,024-source beam scores remain a separate subset comparison. Absolute scores with multiple references are not directly comparable to single-reference benchmarks.

**Pretraining contamination limitation:** OPUS may contain Tatoeba sentences or translations. Helsinki may therefore have seen this held-out test set during pretraining, including in zero-shot evaluation. This has not been ruled out. Matching our train/test split does not remove pretrained-model contamination. Treat Helsinki as an operational portfolio comparison, not evidence of clean generalization or a controlled architectural advantage. A newly collected, independently audited holdout is needed for that claim.

## Results

The fast results below are **user-reported from Colab**, not independently reproduced from raw result artifacts in this workspace. The loss was still falling at epoch 12 (validation 3.77 → 2.06), with a small train/validation gap, suggesting more optimization/capacity may help; this alone does not prove the cause of every error.

| Model / policy | Full-test BLEU | Full-test chrF | Train time | Evidence |
| --- | ---: | ---: | ---: | --- |
| Scratch fast / greedy | 54.53 | 67.93 | 447 s, 12 epochs | User report; T4, ~85k target tokens/s, 1.67 GB peak |
| Scratch large / beam 4 + greedy | TBD | TBD | TBD | New configuration; user GPU run pending |
| Helsinki zero-shot / beam 4 + greedy | TBD | TBD | 0 training | Real pretrained evaluation pending |
| Helsinki fine-tuned / beam 4 + greedy | TBD | TBD | TBD | User GPU run pending |

On the **same 1,024-source subset**, user-reported fast greedy is BLEU 56.61 / chrF 69.77; beam 3 is 57.11 / 70.38. These are not full-test beam results. Keep the actual `metrics.json` files for automatic comparison; no GPU result JSON has been fabricated from this summary. [Reported fast run](docs/user_reported_fast_run.md).

## Models and configurations

| Preset | Width / heads / layers per stack | FFN | Vocabulary | Token cap | Epochs / patience | Padded token budget |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `transformer_colab_fast.yaml` | 256 / 4 / 3 | 1024 | 8k BPE | 24 | 12 / 3 | 16,384 |
| `transformer_colab_large.yaml` | 512 / 8 / 6 | 2048 | 16k BPE | 64 | 30 / 5 | 32,768 |
| `helsinki_colab.yaml` | Pretrained Marian | Pretrained | Pretrained | 64 | 3 / 2 | Batch 64 × accumulation 2 |

Scratch training: post-norm residual blocks, ReLU, dropout, label smoothing, Adam/Noam warmup, clipping, FP16 AMP and early stopping. SentencePiece is fitted on train only. Length buckets minimize padding; two persistent spawned workers, CUDA pinning and nonblocking transfers feed batches. Checkpoints contain optimizer/scaler/scheduler/RNG, best weights, tokenizer and config/data hashes. Only load trusted checkpoints.

Helsinki: preserve its pretrained tokenizer, AdamW 3e-5, 5% linear warmup, label smoothing .1, weight decay .01, full validation **loss only**, length-grouped batches and FP16. Trainer owns optimizer/scheduler/scaler/RNG/early-stopping resume. A run manifest pins the resolved base-model revision and rejects incompatible data/config changes. Zero-shot is completed before any fine-tuning. No test score selects a checkpoint.

## Local setup and CPU validation

Use Python 3.12 and run commands from the checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU wheels; omit these two commands in a CUDA environment.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-helsinki.txt
python -m pytest -q
python -m src.train --config configs/transformer_large_cpu.yaml

# Offline API fixture: random tiny Marian, NOT pretrained Helsinki weights.
python -m scripts.make_marian_fixture --output /tmp/marian-fixture
python -m src.helsinki --config /tmp/marian-fixture/fixture.yaml \
  --stop-after-epochs 1 --skip-test
python -m src.helsinki --config /tmp/marian-fixture/fixture.yaml --resume latest
```

Scratch-only dependencies remain in `requirements.txt`; with only those installed run `pytest tests/test_pipeline.py`. Full pytest requires `requirements-helsinki.txt`. Torchvision is pinned to match torch when Colab has a preinstalled version; it is not part of the translation model. If Colab asks for a runtime restart after package installation, restart before importing Transformers.

CPU smoke uses real split subsets but deliberately tiny models/short runs. Its scores are not quality evidence. The local Marian fixture tests the real Marian tokenizer/model, `Seq2SeqTrainer`, checkpoint resume, generation and result format without network access. Downloading actual Helsinki weights was blocked by the cloud proxy (`huggingface.co` CONNECT 403); real pretrained behavior and Colab/Drive still need the user run. [Latest validation](docs/large_and_helsinki_validation.md).

## Colab A: larger scratch Transformer

Open [notebooks/train_transformer.ipynb](notebooks/train_transformer.ipynb), select T4, run top to bottom:

1. Clone / fast-forward clean `feat/scratch-transformer`; install `requirements.txt`.
2. Mount Drive; choose **new** `scratch_large_v3_seed42` run. Default config is `transformer_colab_large.yaml`.
3. Inspect raw data, train one real epoch without test evaluation, and inspect time/memory/FP16 logs.
4. Print projected total time; resume the **same** large checkpoint for the remainder.
5. Automatically evaluate best validation-loss weights: full greedy, full beam 4, and full beam 4 with trigram suppression.
6. Inspect metrics, both sample files, train log and ETA JSON. On reconnect retain the run name and set `RESUME=True`.

Initial planning range **60–150 T4 minutes including evaluation**, extrapolated from the user's fast run, not measured. A larger model, vocabulary, longer cap and Drive I/O change this. Check first-epoch timing rather than extrapolating CPU/GPU ratios. Memory fit of the 32k-token budget is unverified: on OOM reduce it in a copied YAML under a new run name. Fast-model weights cannot resume into the new architecture/tokenizer.

Equivalent CLI:

```bash
python -u -m src.train --config configs/transformer_colab_large.yaml --stop-after-epochs 1 --skip-test
python -u -m src.train --config configs/transformer_colab_large.yaml \
  --resume checkpoints/transformer_colab_large/last.pt
python -m src.evaluate --checkpoint checkpoints/transformer_colab_large/last.pt
python -m src.translate --checkpoint checkpoints/transformer_colab_large/last.pt \
  --text "Sometimes cows are killed." --beam-size 4
```

## Colab B: Helsinki zero-shot → fine-tune

Open [notebooks/finetune_helsinki.ipynb](notebooks/finetune_helsinki.ipynb), use a **separate** run directory:

1. Clone/update, install `requirements-helsinki.txt`, verify CUDA, mount Drive.
2. Inspect the exact shared split; use the pretrained tokenizer without fitting on any split.
3. Evaluate **zero-shot** full greedy + full beam 4; save results and pin the resolved model revision.
4. Fine-tune one epoch with the planned three-epoch schedule, saving to Drive; skip fine-tuned test evaluation here.
5. Print the time projection (measured epoch plus zero-shot generation cost); resume `latest` for the remaining epochs.
6. Evaluate best validation-loss weights on the same full test set, inspect both stages' files, and generate the comparison table.

The default three epochs / effective batch 128 target a **1–2 hour T4 budget**, not a measured guarantee. Download, full-vocabulary softmax, generated lengths and Drive writes affect it. The notebook measures a real epoch and warns through the displayed projection. If OOM, reduce per-device batch and increase accumulation in a **new** experiment; do not silently alter a resumed optimization schedule.

```bash
python -u -m src.helsinki --config configs/helsinki_colab.yaml --stage zero-shot
python -u -m src.helsinki --config configs/helsinki_colab.yaml --stage fine-tune \
  --stop-after-epochs 1 --skip-test
python -u -m src.helsinki --config configs/helsinki_colab.yaml --stage fine-tune --resume latest
# Or a complete fresh run: python -m src.helsinki --config configs/helsinki_colab.yaml
```

Keep the whole Drive run: manifest, Trainer checkpoint directories, results and final model/tokenizer. Only weights are insufficient for resume. Do not run concurrent writers. Epoch checkpoints replay an interrupted epoch; no mid-epoch recovery is promised. Trainer's safetensors logs may list omitted tied embedding/position aliases when reloading Marian; exact full-model CPU resume is tested. Framework-internal `Trainer.tokenizer` deprecation messages are harmless; this code supplies `processing_class`.

## Result artifacts and comparison

Each scratch run and each Helsinki stage directory (`zero_shot/`, `finetuned/`) writes:

- `metrics.json`: test BLEU/chrF, full greedy/beam scores, signatures, test identity, full config and training/evaluation seconds. Primary scores use unconstrained beam for large/Helsinki. Zero-shot training time is 0.
- `train_log.csv`: per-epoch train/val losses and timing; zero-shot writes a header with no fake training epochs. Scratch loss is token-weighted; Helsinki train loss averages Trainer's logged step losses, so losses across model/tokenizer families are not directly comparable.
- `samples.txt`: first 30 test sources, all available references and primary output (fewer only for a custom small test set).

Large scratch additionally writes `repetition_ablation` scores and `samples_no_repeat.txt`. Timing estimates are in `runtime_estimate.json`. Checkpoints and generated outputs are ignored; `data/raw/` is not ignored.

```bash
python scripts/compare_results.py results/transformer_colab_large results/helsinki --output results/comparison
# In Colab, supply the corresponding Drive results directories instead.
```

The script writes Markdown + CSV, refuses known test-reference/hash/seed/metric-signature mismatches, labels older results without exact fingerprints **legacy/unverified**, separates decoding policies, and omits partial beam scores from the full-test table. CPU smoke files are excluded unless `--include-cpu`. Different tokenizers, token caps, pretraining and beam normalization mean this is a model-system comparison, not a controlled one-variable ablation.

## Design decisions

- Keep the same NFC/whitespace normalization, conservative script/control/length filters and exact-pair deduplication. Preserve original attribution. Group English variants before splitting; references never change across families. Every new full evaluation records a hash of the sorted unique test source/reference pairs.
- The user-reported fast run still improved at the final epoch and had little train/val divergence. Increase capacity and optimization budget rather than increasing regularization: **512/6/2048, 30 epochs, patience 5; retain dropout .1**. This is a hypothesis to test, not a guarantee that all errors were underfitting.
- **16k BPE** on 327k short training pairs balances less fragmentation for rare English/Russian words against sparse vocabulary and output-softmax cost. Keep subword fallback; vocabulary growth does not guarantee correct rare-word meanings. No val/test text fits SentencePiece.
- **Cap 64** instead of 24 addresses possible tail-clause truncation. Measured train-only 16k BPE p99 is 19 on either side; only **one** training pair exceeds 64. Missing clauses can still arise from attention/decoding, so inspect the output rather than assuming the cap solves them.
- **32,768 padded tokens**, max 2,048 sentences and 8,192-example buckets exploit the observed T4 headroom without blindly multiplying a 1.67 GB figure (parameters, optimizer, vocabulary and attention costs change). Measured epoch-zero large batches: **279**, padding **17.9%**, **2,755,161 target tokens/epoch**. Full 30-epoch target exposure is about 82.7M tokens. This does not establish the actual GPU peak; inspect epoch one.
- **Noam warmup 500 / factor .5**: 500/279 ≈1.79 warmup epochs versus fast 800/479≈1.67; estimated target exposure during warmup remains ~5M tokens for both. The peak LR becomes `.5 × 512^(-.5) × 500^(-.5) ≈ 0.000988`, versus fast ≈0.00221. Deeper post-norm training warrants a more conservative peak; larger batches do not require blind linear LR scaling. This factor is configurable and checkpointed through the effective config.
- Preserve length penalty **.6** for scratch to avoid simultaneously changing every decoding knob. Default repetition suppression is **off**: repeated Russian trigrams can be legitimate. Full beam-4 on/off (`no_repeat_ngram_size=3`) is a declared diagnostic ablation; primary scores/samples always stay off. Use validation/manual errors for future policy choices, not the larger of two test scores. Hugging Face's native beam normalization differs from our GNMT-style formula despite a similarly named setting.
- Helsinki's short-sentence dataset uses pretrained tokenizer, cap 64, three epochs, AdamW 3e-5, warmup 5%, batch 64 × accumulation 2 and patience 2. Full validation loss selects the best checkpoint; no generated test score drives training. Pin the resolved pretrained commit from zero-shot for later loads. Model downloads are required, but no dataset download is introduced.
- Use spawned workers and a CPU-only Marian collator that shifts padded labels without holding a CUDA model. This avoids sending the pretrained model into worker processes; Transformers/Accelerate's underlying DataLoader is explicitly configured for spawn.
- Retain training/evaluation evidence and its provenance: user-reported GPU scores, executed CPU checks and unrun GPU workflows are distinct. Helsinki's potential OPUS/Tatoeba overlap must accompany every published comparison. Repeated experimentation on this test set weakens its status as a final holdout; collect a new audited holdout before stronger generalization claims.
- Each cloud task is already isolated. Use the existing checkout; do not create extra Git worktrees. Keep separate experiment names for incompatible configs. Do not publish checkpoints or credentials in Git.

## After the next GPU runs

Send large metrics/logs/samples (suppression off/on), both Helsinki stage result directories, runtime estimates, exact YAMLs/commit, Helsinki `run_manifest.json`, GPU name, peak memory, completed/best epochs, total wall time and warnings/OOM/NaN/interruption details. Keep checkpoints and tokenizer files on Drive. Then review rare-word, clause-omission, repetition and case errors, compare validation curves and decide the next iteration. Build the final combined notebook only as a separate later step.
