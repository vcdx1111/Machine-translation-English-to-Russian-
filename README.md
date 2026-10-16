# English → Russian machine translation

A PyTorch portfolio project comparing a **hand-written Transformer**, a stable larger Transformer, and **Helsinki-NLP/opus-mt-en-ru zero-shot / fine-tuning**. Scratch attention, positions, encoder/decoder and masks are explicit code; no `nn.Transformer`. GPU learning belongs to the user in Colab; this workspace runs CPU diagnostics. Separate scratch/Helsinki notebooks exist; the final combined notebook remains deferred.

## Data and common evaluation

Tracked `data/raw/rus.txt` contains 363,386 UTF-8 TSV rows: English / Russian / attribution. Supplied provenance: Kaggle / Tatoeba; no exact Kaggle URL was supplied. Preserve the CC-BY 2.0 (France) contributor/sentence credits and review the original license before redistribution. No dataset download or Kaggle API is used.

Cleaning retains 363,385 pairs. Seed-42 source-group splits: **327,013 train / 18,248 val / 18,124 test pairs**. Group casefolded English alphanumerics before splitting so punctuation/case variants and multiple translations stay together. [Data report](docs/data_report.md). All workflows call the same split function and score all available references per exact English string using **sacreBLEU 13a + chrF2**. Schema v2 records signatures, raw-data SHA, seed and `test_split_sha256` of the exact source/reference set. Token caps never remove evaluation references.

**OPUS/Tatoeba contamination limitation:** Helsinki may have seen this held-out data during OPUS pretraining. This has not been ruled out. Identical local splits do not establish clean pretrained-model generalization. Multiple-reference scores are also not interchangeable with single-reference benchmarks. Repeated experimentation weakens this test set as a final holdout; collect an independent audited set before stronger claims.

## User-reported Colab results

These are supplied by the user, not independently reproduced from attached metrics/log artifacts. No GPU result JSON is fabricated from the summaries.

| Run | Full greedy BLEU / chrF | Full beam4 BLEU / chrF | Training |
| --- | --- | --- | --- |
| Original fast | 54.53 / 67.93 | Not measured on full test | 447 s, 12 epochs |
| Helsinki zero-shot | 55.38 / 71.78 | 58.10 / 73.09 | 0 updates |
| Helsinki fine-tuned | 56.41 / 74.52 | 62.68 / 76.77 | 1,901 s, 3 epochs |
| Fresh fast v4 | TBD | TBD | New full-beam control run |
| Stable large v4 | TBD | TBD | New Pre-LN run |

Original fast beam3 **subset** (1,024 sources): 57.11 / 70.38; greedy on that same subset: 56.61 / 69.77. Keep these separate from full-test scores.

The previous Post-LN large run **collapsed**: test BLEU 0, repetitive `Я Я` / `Вы Вы`, best epoch 1, early stopping at 6. Validation loss rose 6.42 → 9.80; peak memory was 9.8 GB and epochs took ~137 seconds. Do not resume it. Helsinki finished successfully, but reported unsupported tails such as `Предлагаю вам съездить в Бостон по адресу:` show a faithfulness problem that high BLEU does not eliminate. Trigram suppression is not a general hallucination fix. [Diagnosis and reported evidence](docs/stability_diagnosis.md).

## Current presets

| Config | Width / heads / layers per stack | FFN / BPE / cap | Normalization / tying | Epochs / patience | Tokens / warmup / Noam factor |
| --- | --- | --- | --- | --- | --- |
| `transformer_colab_fast.yaml` | 256 / 4 / 3 | 1024 / 8k / 24 | Post-LN / untied | 12 / 3 | 16,384 / 800 / 1 |
| `transformer_colab_large.yaml` | 512 / 8 / 6 | 2048 / 16k / 64 | **Pre-LN / tied** | 30 / 5 | **16,384 / 3000 / .4** |
| `helsinki_colab.yaml` | Pretrained Marian | Pretrained / cap 64 | Pretrained | 3 / 2 | Batch 64 × accumulation 2 |

Scratch: Adam β=(.9,.98), epsilon 1e-9, label smoothing .1, clipping 1, dropout .1, FP16 on CUDA. Buckets bound **B × (max source length + max target length)** including padding. Both fresh scratch configs evaluate full-test beam4 and greedy; large also records no-repeat-trigram off/on. Primary scores always use suppression **off**.

## Local checks

Use Python 3.12 from the checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
# CPU only; omit this wheel-index command on GPU.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-helsinki.txt
python -m pytest -q
python -m src.overfit --config configs/transformer_colab_large.yaml --output results/overfit.json
python -m src.train --config configs/transformer_fast_v4_cpu.yaml
python -m src.train --config configs/transformer_large_cpu.yaml
```

Scratch-only dependencies are in `requirements.txt`; full pytest additionally needs the Helsinki requirements. CPU smoke outputs are not model-quality benchmarks. The overfit test uses **the full configured width, layer count and vocabulary**, two short pairs, dropout .1 and smoothing .1. It uses fixed LR 3e-4 for bounded one-batch memorization, not 3,000 production warmup steps. It must substantially reduce loss and exactly greedy-decode both references. The larger loss floor with smoothing and a 16k vocabulary is accounted for.

## Colab: fast, then stable large

[notebooks/train_transformer.ipynb](notebooks/train_transformer.ipynb) clones/updates `feat/scratch-transformer` and runs both in order:

1. Select T4, install packages, mount Drive.
2. Use **new** names `scratch_fast_v4_fullbeam` and `scratch_large_stable_v4`. Set only the unfinished run's `resume=True` after reconnect.
3. Fast: one measured epoch → projection → resume same checkpoint → full beam4/greedy evaluation.
4. Fast cleanup cell: after successful evaluation, keep a compact inference-only `best.pt`; delete `last.pt` and resume state.
5. Large: one measured epoch → projection → resume → full beam4/greedy/repetition-ablation evaluation.
6. Large cleanup cell: keep only compact `best.pt`.
7. Supply the existing Helsinki result directory and generate the **four-way** comparison. No Helsinki retraining is required.

Initial planning: **fast ~10–20 minutes, large ~75–150 minutes**, including evaluation/Drive/logging allowances. The old fast measured 447 seconds and old large 137 seconds/epoch; the new architecture, smaller token batch, more updates and checkpoint I/O can change timing. These new ranges are not measured T4 results. The notebook prints the first/recent actual epoch projection; neither dataset nor epochs is silently dropped.

Step/epoch logs stream live. NaN/inf loss stops immediately; validation loss **>1.25× best AND ≥0.5 above best** stops the run. Eight consecutive overflow skips, or >25% skips after at least 20 attempts, also stop. A few early GradScaler calibration skips alone do not establish collapse. Diverged CLI runs exit nonzero, preserve diagnostics/best, and cannot be resumed/pruned automatically. Completed/pruned runs are skipped on notebook reconnect after config checks.

Equivalent CLI (use a new config/run for incompatible changes):

```bash
python -u -m src.train --config configs/transformer_colab_fast.yaml --stop-after-epochs 1 --skip-test
python -u -m src.train --config configs/transformer_colab_fast.yaml --resume checkpoints/scratch_fast_v4/last.pt
python -u -m src.train --config configs/transformer_colab_large.yaml --stop-after-epochs 1 --skip-test
python -u -m src.train --config configs/transformer_colab_large.yaml --resume checkpoints/scratch_large_stable_v4/last.pt
# Only AFTER completed training and evaluation:
python -m src.checkpoints --checkpoint-dir checkpoints/scratch_large_stable_v4 --results-dir results/scratch_large_stable_v4
# Compact best remains evaluable/translatable, but cannot resume training:
python -m src.evaluate --checkpoint checkpoints/scratch_large_stable_v4/best.pt --results-dir results/large_recheck
python -m src.translate --checkpoint checkpoints/scratch_large_stable_v4/best.pt --text "Sometimes cows are killed." --beam-size 4
```

Cleanup is intentional and irreversible for resume capability: model/optimizer/scaler/RNG/current-state duplicates are removed after verifying matching run IDs and completed evaluation. The best weights, tokenizer, metadata and all results remain. Running/calibrated/diverged runs are refused. Do not run concurrent writers. Retain full state instead if you plan to continue training later. Only load checkpoints you trust.

## Helsinki workflow

[notebooks/finetune_helsinki.ipynb](notebooks/finetune_helsinki.ipynb): clone/install `requirements-helsinki.txt` → separate Drive run → zero-shot full evaluation → first-epoch timing → Seq2SeqTrainer resume → best-validation full evaluation. The completed user run took 1,901 seconds of fine-tuning; prior 1–2 hour planning was conservative, not a measured requirement. The code/config is unchanged by this scratch stabilization work.

The manifest pins the resolved base snapshot, data/test identity and optimization contract. Trainer saves optimizer, linear schedule, scaler/RNG/early-stopping state; keep the whole run for resume. Validation uses loss only. Decoder uses pretrained tokenizer, deterministic cached generation, beam4 and greedy. The fixture command below tests the API offline, **not pretrained weights**:

```bash
python -m scripts.make_marian_fixture --output /tmp/marian-fixture
python -m src.helsinki --config /tmp/marian-fixture/fixture.yaml --stop-after-epochs 1 --skip-test
python -m src.helsinki --config /tmp/marian-fixture/fixture.yaml --resume latest
```

Cloud access to actual Hugging Face weights was blocked by CONNECT 403; the user GPU results are separate evidence. CPU tests use real Marian/Trainer with local random weights and verify exact resume. Torchvision is pinned to match torch for Colab compatibility. Framework tied-alias reload/tokenizer-deprecation messages do not alter the CPU equality test. Unsupported extra tails require source-faithfulness review; no unvalidated Helsinki decoding change is silently introduced here.

## Artifacts and fair comparison

Each scratch run / Helsinki stage (`zero_shot/`, `finetuned/`) writes `metrics.json`, `train_log.csv`, `samples.txt`. Schema v2 includes exact test identity, signatures, effective config and full-test greedy/beam scores. Zero-shot training time is 0 and its CSV is header-only.

Scratch adds `step_log.csv`: every attempted batch's **unscaled pre-clip norm**, clipping bound, proposed LR, **actual applied LR** (blank on skipped update), successful optimizer-step index, scaler before/after and update/skip event. `run_status.json`, `divergence.json` when applicable, and `runtime_estimate.json` distinguish completion, early stopping, calibration and failure. Large suppression-on output goes to `samples_no_repeat.txt` and `repetition_ablation`. Each new scratch run has an ID to prevent cross-run checkpoint cleanup. Old evaluation checkpoints remain loadable, but training pipeline v3/new architecture require new runs.

```bash
python scripts/compare_results.py results/scratch_fast_v4 results/scratch_large_stable_v4 results/helsinki \
  --require-four-models --output results/comparison_v4
```

Use corresponding Drive paths in Colab. The strict flag requires fast, large, Helsinki zero-shot and fine-tuned with schema v2 exact identities and full beam4 plus greedy (CPU fixtures are explicitly exempted with `--include-cpu`); checks test hashes/signatures/seed/counts, shows status/experiment/policy, labels legacy identity as unverified, and excludes subset scores and CPU smoke by default. Times are per-run totals repeated per decoding row; do not sum them. It cannot reconstruct raw GPU artifacts from chat summaries. Different tokenizers/caps/pretraining and beam normalization make this a system comparison, not a single-variable capacity ablation.

## Design decisions

- Preserve data cleaning, grouping, split seed and all test references. No test text fits scratch SentencePiece. Source paraphrases and pretrained OPUS overlap remain limitations.
- Diagnosis: old layers use **Post-LN**. The 500-step Noam plan implied LR ≈0.000551 at step 279 and ≈0.000935 at step 558, consistent with the reported first two epoch LRs. This supports an overly aggressive schedule/depth interaction, not a scheduler-arithmetic bug. Without old stepwise norms we cannot prove a unique cause. Two initial FP16 skips can be ordinary scale calibration.
- **Pre-LN is optional** and defaults on for large, with final encoder and decoder normalization. Fast explicitly retains Post-LN as the successful control. This is a combined stabilization intervention, not proof that one change alone fixes convergence.
- Keep `sqrt(d_model)` embedding scaling. Legacy Xavier initialized the embedding using both vocabulary and width (16k/512 std≈.011); after scaling its content std≈.25 versus positional components. Large now uses **Normal(0,d_model^-1/2)**, producing unit content variance after scaling; linear weights stay Xavier, biases zero, LayerNorm scale/bias 1/0. Fast keeps legacy initialization to preserve its control recipe.
- **Tie large input/output weights** for the joint vocabulary; fast stays untied. This removes one 16k×512 output matrix (~8.2M parameters). PAD masks/ignored target loss remain explicit; the tied PAD row can receive output-layer gradients. Initialization zeroes its PAD row but does not claim it is permanently frozen.
- Large keeps 512×6 / FFN2048 / dropout .1 / 16k BPE / cap64. Lower `max_tokens` **32,768→16,384**, max sentences 1,024. Train-only 16k tokenization still clips only one training pair at 64. Measured new epoch-zero batches **479** versus old 279; padding **11.9%**. Smaller batches increase update frequency and reduce the observed 9.8 GB footprint; new T4 peak is not yet measured.
- Warmup **3000** / factor **.4** yields max LR `0.4/sqrt(512*3000)≈0.00032275`, ~one third of the previous 0.000988. With 479 batches it spans **6.26 epochs** versus old ~1.79. Early/end-of-epoch LR is not the mathematical peak. Successful optimizer updates alone advance Noam, so scaler skips cannot accelerate warmup.
- AMP update ordering: scale → backward → **unscale → inspect/clip norm**. Nonfinite gradients explicitly skip both optimizer and Noam and back off the scaler. This also covers the edge case where all gradient elements are finite but FP32 norm reduction overflows: relying on `scaler.step` alone could miss that. Norm/scale/LR logging makes the next T4 diagnosis auditable. CPU GradScaler tests cover both overflow modes; they do not claim GPU numerical equivalence.
- Stop nonfinite loss immediately and stop substantial validation regression before another epoch. Preserve the last safe/best checkpoints; never save corrupted weights as best. Failure remains labeled even if older best weights are evaluated. Guards use validation/training only, never test metrics.
- Full-width/depth/vocabulary one-batch overfit tests require a large relative loss drop and exact greedy decoding. Smoothing .1 has an entropy floor (~1.293 for 16k classes), so requiring smoothed loss below 1 would be mathematically wrong. The bounded test uses LR 3e-4 and does not certify production warmup, FP16 or generalization.
- Preserve full-test unconstrained beam4 as primary, greedy alongside, large trigram suppression on/off as a diagnostic. Repetition suppression cannot establish semantic fidelity or fix arbitrary Helsinki extra tails. Avoid selecting whichever test-policy score is highest.
- Save full resume state while training; compact to best-only after evaluation to limit Drive use. Unique run IDs, status checks and model/data checks protect unrelated files. Keep logs, samples, tokenizer and exact config. CPU toy results, user-reported GPU observations and unrun new GPU workflows remain separate.
- Each cloud task is isolated; use the existing checkout, no extra worktrees. Never commit model weights or credentials.

See [stability diagnosis and validation](docs/stability_diagnosis.md). After running, send metrics, train/step logs, status/ETA, divergence details if any, samples (large off/on), exact YAML/commit, GPU memory/epoch time and errors. Keep compact best on Drive. Build the final all-stage notebook later.
