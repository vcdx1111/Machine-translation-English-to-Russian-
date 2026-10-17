# English → Russian machine translation

A portfolio study of what a Transformer built from scratch can learn from a small parallel corpus, how training can fail, and how it compares with a pretrained translation model. The PyTorch implementation contains its own attention, positional encoding, encoder, decoder and masks; it does not use `nn.Transformer`. A separate Helsinki Marian baseline is evaluated before and after fine-tuning.

**Final outcome:** the stable scratch model reaches **59.88 BLEU / 72.37 chrF** with beam 4; fine-tuned Helsinki reaches **62.68 / 76.77** on the same held-out sources. Training is complete. The integrated notebook defaults to reading the saved results; reproducing training is an explicit opt-in.

## Data and evaluation

The checked-in `data/raw/rus.txt` comes from Kaggle, **“English-Russian dictionary for machine translate”**, originally from [Tatoeba](https://tatoeba.org/). It contains **363,386 UTF-8 TSV rows**, each with **English / Russian / attribution**. The corpus is Tatoeba **[CC-BY 2.0](https://creativecommons.org/licenses/by/2.0/)**; row credits identify the France port, [CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/). Original contributor attribution is preserved in the raw file and processed rows. The code's MIT license does not relicense the data. The exact Kaggle listing URL was not supplied; the title is recorded instead of guessing a URL.

NFC normalization and whitespace collapse preserve case, punctuation and stress accents. Cleaning removes empty/invalid-script/control-character pairs, duplicate normalized pairs and pairs longer than 300 characters or 60 whitespace words on either side. Only one pair is removed by these rules. See the measured [data report](docs/data_report.md).

Splitting individual translation pairs would leak English sentences with several Russian answers. Instead, casefolded English alphanumeric keys define groups: all translations and simple punctuation/case variants go to **one** of train/validation/test. Sorted keys are shuffled with seed 42 and split **90/5/5 by group**, producing **327,013 / 18,248 / 18,124 pairs**. The test set has **12,556 distinct exact English strings** (12,542 normalized groups). SentencePiece is fitted on train only.

Evaluation groups all available Russian references for each exact English source. Metrics use sacreBLEU 2.5.1, BLEU tokenizer `13a`, variable references (`nrefs:var`) and default chrF2. Missing reference slots are `None`, not duplicated translations. Full-test greedy and beam-4 use exactly the same sources and references. Schema v2 records the config, raw-data hash, seed and exact reference-pair hash:

```text
test_split_sha256 = bc6e2388f0938c3332e747f174c98d16530f69b4641f0cb6ddb3c074e8f39f30
```

## Final Colab T4 results

These are the user's completed GPU measurements, recorded in [final_results.json](docs/final_results.json). Original metrics and samples live in Google Drive; the numbers below were not produced by the CPU smoke tests.

| Model | Greedy BLEU | Greedy chrF | Beam-4 BLEU | Beam-4 chrF | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer fast v4, d256, 3 layers, Post-LN | 54.53 | 67.93 | 56.08 | 69.18 | 475 s (7.9 min) |
| Transformer large v4, d512, 6 layers, Pre-LN | 58.65 | 71.44 | 59.88 | 72.37 | 4,446 s (74.1 min) |
| Large v4 + no-repeat-3 | — | — | 59.87 | 72.36 | Same weights |
| Helsinki opus-mt-en-ru, zero-shot | 55.38 | 71.78 | 58.10 | 73.09 | 0 s |
| Helsinki opus-mt-en-ru, fine-tuned, 3 epochs | 56.41 | 74.52 | **62.68** | **76.77** | 1,901 s (31.7 min) |

Times describe reported training, excluding installation, downloads and final generation. All rows use the full test set; the older 1,024-source beam experiment is excluded.

Increasing scratch capacity improves beam BLEU by **3.80**. Large v4 beats Helsinki zero-shot in BLEU (**59.88 vs 58.10**) but trails it in chrF (**72.37 vs 73.09**). This is consistent with stronger adaptation to Tatoeba phrasing and weaker Russian morphology; chrF alone does not establish the cause. Beam search adds **1.55 / 1.23 / 6.27 BLEU** to fast / large / fine-tuned Helsinki. The greater Helsinki gain indicates decoding matters substantially for this model. No-repeat-3 changes large BLEU by **−0.01** and is left off. Fast v4 reproduces the earlier run's greedy BLEU to two decimals, evidence of score-level fixed-seed repeatability.

Large v4's validation loss **converged**: about **2.155 → 2.144 after epoch 20**. Its best checkpoint was epoch 25, and patience was exhausted at epoch 30 (`early_stopped`). This run completed all 30 planned epochs; early stopping did not shorten it.

## Translation examples and remaining errors

These are verbatim **large v4** outputs reported during evaluation. Exact fast/zero-shot/fine-tuned strings for these sources were not supplied. “Reported correct” is a human observation, not a reconstructed model output; a dash means unavailable. This distinction prevents invented cross-model examples.

| English source | Fast v4 | Large v4 | Helsinki zero-shot | Helsinki fine-tuned |
| --- | --- | --- | --- | --- |
| The plans were discarded. | — | Планы были на борт. | — | Reported correct; exact output unavailable |
| Sometimes cows are killed by coyotes. | — | Иногда коров убивают коров. | — | Reported correct; exact output unavailable |
| They have lost 10 games in a row since their winning streak ended. | — | Они проиграли 10 игры в весе с тех пор, как выигрывают дуба. | — | Reported correct; exact output unavailable |
| She quickly put the money out of sight. | — | Она быстро убрал деньги из виду. | — | Reported correct; exact output unavailable |
| Tom is clearly lost. | Reported correct; exact output unavailable | Том явно потерял. | — | Reported correct; exact output unavailable |

The examples expose rare-word substitution, long-sentence semantic collapse, gender agreement errors and occasional regressions relative to the smaller model. Fine-tuned Helsinki also sometimes adds unsupported content, such as **“Предлагаю вам съездить в Бостон по адресу:”**. Higher aggregate scores do not guarantee faithfulness.

`scripts/build_example_table.py` joins actual saved outputs by English source, includes common available examples from all four runs and writes `docs/translation_examples.md`. Standard `samples.txt` contains only 30 examples, so the error cases above may be absent. If absent, use existing full prediction artifacts or saved best models for **inference only**; further training is unnecessary. The integrated notebook includes an optional saved-checkpoint inference section to obtain these exact four-way outputs without training.

## Architecture and training

Both scratch models use sinusoidal positions, scaled dot-product multi-head attention, causal/padding masks, residual connections, feed-forward blocks, dropout, label-smoothed cross-entropy, Adam (`β=(0.9,0.98)`, `ε=1e−9`), Noam warmup, gradient clipping at 1.0 and CUDA FP16 GradScaler. Validation uses loss and a fixed 128-source greedy subset; final evaluation uses full-test batched beam-4 plus greedy. Step logs expose actual learning rate, unscaled gradient norm, scaler and skipped updates. Nonfinite losses/gradients and sharp validation regression stop a failed run.

| Setting | Fast v4 | Stable large v4 |
| --- | --- | --- |
| Joint train-only SentencePiece BPE vocabulary | 8,000 | 16,000 |
| d_model / heads / encoder + decoder depth | 256 / 4 / 3 + 3 | 512 / 8 / 6 + 6 |
| Feed-forward size / dropout | 1,024 / 0.1 | 2,048 / 0.1 |
| Layer normalization | Post-LN | Pre-LN + final norms |
| Embedding / output weights | Shared EN/RU input; untied output | Shared EN/RU input and output |
| Initialization | Legacy Xavier | Xavier matrices, embedding std d_model^−0.5, zero linear biases |
| Token cap including BOS/EOS | 24 | 64 |
| Padded source + target tokens/batch | 16,384 | 16,384 |
| Epochs / patience | 12 / 3 | 30 / 5 |
| Noam warmup / factor | 800 / 1.0 | 3,000 / 0.4 |
| Beam / length penalty / no-repeat | 4 / 0.6 / off | 4 / 0.6 / off; separate 3-gram ablation |

Helsinki uses its own pretrained Marian tokenizer, max source length 64, max 64 new target tokens, Seq2SeqTrainer for 3 epochs, batch 64 × accumulation 2, AdamW learning rate **3e−5**, 5% warmup, weight decay 0.01, smoothing 0.1, gradient clip 1.0, FP16 and loss-based best selection. Zero-shot evaluation precedes any fine-tuning, and the resolved pretrained revision is recorded by the modular runner.

## Failed experiment: large v3 → stable v4

| Setting / observation | Large v3 (failed) | Large v4 (stable) |
| --- | --- | --- |
| Normalization | Post-LN | Pre-LN + final norms |
| Input scaling / initialization | sqrt(d_model) scaling already present; legacy Xavier embedding | Keep scaling; normalize embedding initialization |
| Embedding/output tying | Off | On |
| Padded token budget | 32,768 | 16,384 |
| Noam warmup / factor | 500 / 0.5 | 3,000 / 0.4 |
| Peak learning rate | ~9.4e−4 reported | ~3.1e−4 reported (formula: 3.23e−4) |
| Train loss | Stalled near 6.4 | 6.64 → 1.63 |
| Validation loss | 6.42 → 9.80 | 5.28 → 2.14 |
| Training outcome | Failed by epoch 6; best epoch 1 | Best epoch 25; validation plateau by epoch 30 |
| Test output | BLEU 0; “Я Я” / “Вы Вы” | Beam BLEU 59.88 |
| FP16 skipped steps | 2 in epoch 1 | 2 across 30 epochs |

A high learning-rate peak reached too soon in a deep Post-LN stack is consistent with the observed instability. Doubling the earlier token budget also reduced optimizer steps per epoch. The v4 changes make optimization gentler; several settings changed together, so this is a stabilization experiment rather than a causal ablation. FP16 skips identify nonfinite gradients; two isolated skips do not by themselves explain collapse. The scaler now unscales before norm/clipping, skips both optimizer and scheduler on nonfinite gradients and logs the scale. The repeated v3 outputs were symptoms of training failure; the no-repeat ablation does not support treating them as a decoding fix.

## Run locally or in Colab

Python 3.11/3.12 is recommended. From a fresh checkout:

```bash
git clone --branch feat/scratch-transformer https://github.com/vcdx1111/Machine-translation-English-to-Russian-.git
cd Machine-translation-English-to-Russian-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-helsinki.txt
python -m pip check
pytest -q
```

The requirements files pin direct dependencies. For a CPU-only machine, install the matching CPU torch/torchvision wheels from PyTorch's CPU index before the requirements. No Kaggle API or dataset download script is needed; raw TSV data is tracked.

For **optional** reproduction, the modular CLIs are:

```bash
python -m src.train --config configs/transformer_colab_fast.yaml
python -m src.train --config configs/transformer_colab_large.yaml
python -m src.helsinki --config configs/helsinki_colab.yaml --stage all
python -m src.evaluate --checkpoint checkpoints/scratch_large_stable_v4/best.pt --beam-size 4
```

Use a distinct run directory when repeating an experiment. Modular training can resume epoch-boundary optimizer/RNG state from `last.pt` or Helsinki `--resume latest`. `python -m src.checkpoints --help` explains irreversible best-only compaction; compact artifacts support inference, not training resume. Never load untrusted PyTorch pickle checkpoints.

For Colab, select a **T4** runtime and open:

- [en_ru_translation_full.ipynb](notebooks/en_ru_translation_full.ipynb): standalone code, no `src` imports. Run top-to-bottom to mount Drive, inspect data, load final results and compare. All model training flags default **False**. `RUN_FAST`, `RUN_LARGE`, `RUN_HELSINKI_ZERO_SHOT` and `RUN_HELSINKI_FT` explicitly opt into reproduction. Only best checkpoints are saved; this notebook trades optimizer-state resume for Drive space.
- [train_transformer.ipynb](notebooks/train_transformer.ipynb): modular fast/large experiment runner, resumable training, streaming logs and checkpoint cleanup.
- [finetune_helsinki.ipynb](notebooks/finetune_helsinki.ipynb): modular zero-shot/fine-tuning runner, resumable Trainer state and streaming logs.

The modular notebooks clone `feat/scratch-transformer`; the standalone notebook fetches that branch **only for the checked-in data** if necessary, and runs embedded code. When reusing a Colab session with a different torch/transformers version already imported, restart the runtime after installation before running model cells.

Saved artifacts live under `/content/drive/MyDrive/en_ru_translation/`:

```text
scratch_fast_v4_fullbeam/results/
scratch_large_stable_v4/results/
helsinki_v1_seed42/results/zero_shot/
helsinki_v1_seed42/results/finetuned/
```

Each run normally writes schema-v2 `metrics.json`, `samples.txt` and `train_log.csv`. The recovered Helsinki runs have **metrics and samples only**: both completed successfully, but their missing training logs cannot be reconstructed. The comparison script labels legacy Helsinki evaluation artifacts `completed`, preserves explicit statuses and does not require a training log.

### Generate figures and exact example tables without training

Run these in the cloned repository in Colab after mounting Drive:

```bash
python scripts/compare_results.py \
  /content/drive/MyDrive/en_ru_translation/scratch_fast_v4_fullbeam/results \
  /content/drive/MyDrive/en_ru_translation/scratch_large_stable_v4/results \
  /content/drive/MyDrive/en_ru_translation/helsinki_v1_seed42/results \
  --require-four-models --output docs/model_comparison
python scripts/plot_results.py \
  --results-root /content/drive/MyDrive/en_ru_translation \
  --loss-log /content/drive/MyDrive/en_ru_translation/scratch_large_stable_v4/results/train_log.csv \
  --output docs/figures
python scripts/build_example_table.py \
  --results-root /content/drive/MyDrive/en_ru_translation
```

The plots are `docs/figures/model_scores.png` and `large_v4_loss.png` with a provenance JSON. The score chart can also be generated locally with `python scripts/plot_results.py` from the explicitly user-reported aggregates. A loss curve **always** requires the real epoch CSV; endpoint values are never expanded into a synthetic curve. After generation, commit the figures and generated Markdown (including verbatim examples), and embed the desired images in this README. No broken image links are committed in advance.

## Design decisions

- **Keep evaluation leakage-resistant:** group source variants before splitting; retain multiple references and store exact test identity. Token caps truncate model inputs/targets, not the held-out reference set.
- **Use dataset-sized tokenizers:** 8k BPE keeps the short-sentence baseline cheap; 16k reduces segmentation of rare Russian words for the larger model. Dropout 0.1 remains conservative with 327k training pairs. Neither vocabulary choice was isolated in a controlled ablation.
- **Reduce padding first:** shuffle length-sorted pools, then budget padded source+target tokens. CLI loaders use two spawned workers and CUDA pinned memory. Embedded notebook classes cannot be imported by spawned workers, so its loaders use zero workers while retaining the same batches and optimization settings.
- **Cap lengths using observed distributions:** English/Russian p99 lengths are 13/12 whitespace words; the fast 24-subword cap comes from the measured tokenizer profile. Large's 64 cap protects longer examples. Word and subword counts are different units; short training data still limits long-sentence generalization.
- **Stabilize large optimization:** with 16,384 tokens and roughly 479 batches/epoch, 3,000 warmup steps span about 6.3 epochs. `0.4 × 512^−0.5 × 3000^−0.5 ≈ 3.23e−4`, below 5e−4; warmup advances only after successful updates. This preserves smaller updates while using T4 memory comfortably. Pre-LN, normalized embedding initialization and weight tying reduce the fragile scale interaction in v3.
- **Separate quality from execution status:** an evaluated run can be `early_stopped`; convergence and best epoch explain that status. Legacy Helsinki artifacts confirm completed evaluation despite missing CSV logs. CPU fixtures carry `cpu_smoke_only` and never enter GPU score tables.
- **Make final publication honest:** final aggregates are attributed to user measurements; unavailable verbatim outputs and logs stay marked missing. No extra GPU training is part of finalization. Integrated training is disabled by default and best-only checkpoints avoid retaining optimizer duplicates. Overwriting Drive files can still accumulate previous versions in Trash; review and empty Trash separately if needed.
- **Retain experiment history:** earlier configs, profiling and stability reports remain under their descriptive paths or `configs/archive` / `docs/archive`. Their historical estimates are superseded by the final measured table above.

## Limits and future work

Multiple valid references increase the chance that a hypothesis matches one answer, so these BLEU scores can be higher than single-reference scores. Tatoeba mostly contains short, comparatively simple sentences; this test does not establish quality on news, technical documents or long text. Normalized-source grouping prevents direct English duplication but cannot exclude semantic paraphrases or domain overlap.

Helsinki's OPUS pretraining **may include Tatoeba**, so test contamination has not been ruled out. Its comparison includes a potential pretraining advantage as well as domain adaptation. These numbers **cannot be compared directly with WMT or other general benchmark scores**. Only one split/seed is reported; confidence intervals and multi-seed variance were not measured. Rare vocabulary, Russian inflection, long-context retention and hallucinated suffixes remain weaknesses.

Future work could evaluate an independently sourced, contamination-audited test set, add paired bootstrap intervals, inspect faithfulness/morphology systematically and improve domain or long-sentence coverage. Further training is outside this completed experiment.

## Repository and validation

`src/` holds reusable pipeline code; `configs/` records reproducible experiments; `tests/` checks masks/shapes/tokenizers, optimizer/scaler safeguards, resume, batched decoding, result identity and tiny-batch overfitting. `scripts/` provides reporting, notebook generation and publication audits. `docs/` holds measured reports and finalization evidence. Generated model weights, processed data and run outputs are ignored; the raw corpus and its contributor credits are retained.

```bash
python scripts/build_full_notebook.py --check
python scripts/validate_full_notebook.py --output /tmp/en-ru-notebook-smoke
python scripts/audit_publication.py
pytest -q
```

The standalone smoke executes all stages with tiny CPU models, including a **random local Marian fixture**, and verifies artifacts. It does not validate pretrained Helsinki quality or Colab CUDA behavior. See [final validation](docs/final_validation.md) for executed checks. Code: [MIT](LICENSE). Corpus: Tatoeba CC-BY 2.0 with original row attribution.
