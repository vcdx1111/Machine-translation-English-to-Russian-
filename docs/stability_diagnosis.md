# Post-LN collapse diagnosis and stable v4 validation

> This records the stabilization implementation and CPU evidence. The subsequent completed large v4 GPU results and convergence analysis are in the root README.

## User report and limits of diagnosis

The user reported Helsinki completion: zero-shot greedy 55.38/71.78, beam4 58.10/73.09; fine-tuned greedy 56.41/74.52, beam4 62.68/76.77 after 3 epochs / 1,901 seconds. These are multi-reference BLEU/chrF. A reported unsupported tail (`Предлагаю вам съездить в Бостон по адресу:`) illustrates source-faithfulness risk. No Helsinki weights/decoding policy is changed in this stabilization task. OPUS/Tatoeba pretraining overlap remains unresolved.

Old large failed: train loss 7.03→6.49→6.66→6.48→6.42→6.40; val 6.42→6.48→7.27→9.80→8.95→9.02; best epoch1, early stop6, BLEU0 and repeated `Я Я`/`Вы Вы`. Reported 137 sec/epoch and 9.8GB peak. Actual result artifacts were not attached, so this is user-reported evidence, not independently verified GPU telemetry.

Code confirms Post-LN, sqrt(d_model) scale, untied input/output weights, and Xavier on embeddings plus all weight matrices. For 16k/512 this embedding std is ~.011; scaled content std ~.25. The old Noam arithmetic predicts ~.000551 at step279 and ~.000935 at step558, matching reported epoch LRs. A short warmup/high LR with deep Post-LN is a strong candidate mechanism. Old norms/activation traces are absent; it is not possible to prove LR is the sole cause. Two first-epoch scaler skips alone can be ordinary initial FP16 calibration.

## Chosen changes

Pre-LN plus final encoder/decoder norms; normal embedding std=512^-0.5 before sqrt scaling; Xavier linears with zero bias; tied large input/output weights. Keep fast Post-LN/legacy initialization/untied as a control. New large warmup3000, factor.4, peak ~3.2275e-4; max_tokens16384. All require new v4 runs. Noam clock advances only on finite actual updates.

Actual full training-data profile (no model training):

```json
{
  "batches": 479,
  "pairs": 327013,
  "nonpad_tokens": 6386297,
  "padded_tokens": 7250687,
  "padding_fraction": 0.11921491025609021,
  "target_tokens": 2755161,
  "warmup_epochs": 6.263048016701461,
  "peak_lr": 0.00032274861218395146,
  "lr_first_epoch": 5.153219507870425e-05,
  "vocab_size": 16000,
  "diagnostic": "Train-only 16k tokenizer from prior unchanged-raw profile reused; no GPU training"
}
```

The new 479-step epoch gives warmup6.26 epochs and first-epoch LR ~5.15e-5 (before any scaler skips). Compare previous 279 steps/epoch and warmup1.79 epochs. These are successful-update-count estimates, not T4 timing measurements.

## Overfit diagnostic (CPU, full configured models)

```bash
python -m src.overfit --config configs/transformer_colab_fast.yaml --output results/overfit_fast_v4.json
python -m src.overfit --config configs/transformer_colab_large.yaml --output results/overfit_large_v4.json
```

Two short distinct source/target pairs, full configured vocabulary/width/depth, dropout .1, smoothing .1, max sequence8 and fixed Adam LR3e-4. This tests learnability, not production warmup/FP16 or translation quality. The two outputs must greedily equal both targets; smoothed loss must fall >65% and approach its vocabulary-dependent entropy floor.

```json
{
  "fast": {
    "config": "configs/transformer_colab_fast.yaml",
    "model": {
      "d_model": 256,
      "heads": 4,
      "layers": 3,
      "d_ff": 1024,
      "dropout": 0.1,
      "max_length": 8,
      "norm_first": false,
      "tie_embeddings": false,
      "initialization": "legacy_xavier"
    },
    "vocab_size": 8000,
    "theoretical_smoothed_loss_floor": 1.2236503501033653,
    "steps": 65,
    "initial_loss": 8.918869018554688,
    "final_loss": 1.4625639915466309,
    "loss_ratio": 0.16398536501701433,
    "exact_greedy": true,
    "seconds": 5.242305514999316,
    "passed": true,
    "diagnostic": "Full width/depth/vocabulary, dropout from YAML, label smoothing .1; fixed LR 3e-4 for one-batch memorization, NOT production warmup validation"
  },
  "large": {
    "config": "configs/transformer_colab_large.yaml",
    "model": {
      "d_model": 512,
      "heads": 8,
      "layers": 6,
      "d_ff": 2048,
      "dropout": 0.1,
      "max_length": 8,
      "norm_first": true,
      "tie_embeddings": true,
      "initialization": "scaled_embedding"
    },
    "vocab_size": 16000,
    "theoretical_smoothed_loss_floor": 1.2930368886883228,
    "steps": 10,
    "initial_loss": 10.710299491882324,
    "final_loss": 1.4842365980148315,
    "loss_ratio": 0.13858030759455248,
    "exact_greedy": true,
    "seconds": 3.6459358380016056,
    "passed": true,
    "diagnostic": "Full width/depth/vocabulary, dropout from YAML, label smoothing .1; fixed LR 3e-4 for one-batch memorization, NOT production warmup validation"
  }
}
```

## Pytest

```text
..................................                                       [100%]
34 passed in 23.47s
```

New coverage: full fast/large overfit, Pre-LN causality/padding/tied weights/initialization, real CPU GradScaler with elementwise inf and finite-element norm overflow, skipped optimizer/Noam updates, exact applied LR/scale logging, NaN-loss stop, validation-regression stop with safe best retained, partial-epoch step-log replay, completion-gated/idempotent best-only cleanup and inference/resume refusal, and required four-way comparison. Prior tokenizer/model/Marian resume/notebook/live-streaming tests still pass. GPU FP16 overflow behavior requires the next user run.

## CPU full-pipeline smoke

```bash
python -m src.train --config configs/transformer_fast_v4_cpu.yaml
python -m src.train --config configs/transformer_large_cpu.yaml
```

Both use 96 real train pairs, 16 validation, 40 held-out pairs, reduced width/depth/vocabulary and two epochs. These smoke configs use smaller token/LR budgets; the full-architecture overfit tests above are separate. Full beam4 and full greedy executed; large also did full beam4/no-repeat3. Scores are not portfolio quality evidence. Both result identities match the local Marian smoke from the earlier stage.

Actual fast CSV:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds,checkpoint_seconds,train_seconds,val_seconds,target_tokens_per_second,nonpad_tokens_per_second,padding_fraction,val_greedy_bleu,val_greedy_chrf,val_greedy_seconds,val_greedy_sources,cuda_peak_gb,amp_enabled,logits_dtype,skipped_steps
1,5.587925540600935,5.438295200892857,0.001381067932004976,0.1583015290016192,0.017456260000471957,0.0818066709980485,0.0030833989949314855,22223.126522815826,45473.06417209839,0.10058027079303677,0.0,2.6351813826146477,0.030520781998347957,8,0.0,False,torch.float32,0
2,5.366700293076183,5.179632132393973,0.002762135864009952,0.09092173100361833,0.016992773998936173,0.06395023899676744,0.003450336000241805,28428.353490467744,58170.228264323436,0.10058027079303677,0.0,1.3706140350877194,0.022441578999860212,8,0.0,False,torch.float32,0
```

Actual stable large CSV:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds,checkpoint_seconds,train_seconds,val_seconds,target_tokens_per_second,nonpad_tokens_per_second,padding_fraction,val_greedy_bleu,val_greedy_chrf,val_greedy_seconds,val_greedy_sources,cuda_peak_gb,amp_enabled,logits_dtype,skipped_steps
1,5.9103370211308,5.663277259239783,0.0019531250000000004,0.19891696899867384,0.01472800200281199,0.14002953300223453,0.0029429920032271184,14004.1886733187,28086.932204060402,0.1354143767861068,0.0,0.0,0.02805792099388782,8,0.0,False,torch.float32,0
2,5.484559257075472,5.250951913686899,0.0027950849718747376,0.05390173200430581,0.014592065999750048,0.0460054420036613,0.002674618997843936,42625.39201001341,85489.88616796673,0.1354143767861068,0.0,0.0,0.0043449939985293895,8,0.0,False,torch.float32,0
```

Each step CSV has one row per attempted training batch, with unscaled norm, clip bound, proposed/applied LR, optimizer step, scaler before/after and skip/event. A blank applied LR indicates no update. Mid-epoch divergence writes a diagnostic and retains the prior completed checkpoint. Validation ratio>1.25 plus delta≥.5 stops before the next epoch. CPU tests inject failures deliberately to exercise these paths; they do not disable assertions to pass.

## Checkpoint space and estimated GPU time

Cleanup keeps only a compact inference `best.pt` after matching run IDs and completed evaluation; drops current/optimizer/scaler/RNG duplicates and `last.pt`, keeps all result files and tokenizer. Diverged/running/calibrated runs are refused. Resume from compact best is refused. See the CLI verification appendix below.

Notebook: fast measure→resume→evaluate→cleanup, then stable large measure→resume→evaluate→cleanup, then four-model comparison with existing Helsinki stage results. New names `scratch_fast_v4_fullbeam`, `scratch_large_stable_v4`. No old collapsed checkpoint is reused. Initial T4 planning ranges: fast10–20min, large75–150min, based on user fast447s and old large137s/epoch with allowances for more updates, logging, new norm/init and Drive I/O. Actual first/recent epoch timing replaces these estimates. No new GPU training or numerical-stability claim was made here.

## Final CLI verification

```bash
python -m src.checkpoints --checkpoint-dir checkpoints/fast_v4_cpu --results-dir results/fast_v4_cpu
python -m src.checkpoints --checkpoint-dir checkpoints/stable_large_cpu_v4 --results-dir results/stable_large_cpu_v4
python -m src.evaluate --checkpoint checkpoints/stable_large_cpu_v4/best.pt --results-dir results/stable_v4_recheck
```

Observed checkpoint-byte reports (model checkpoint files only, not tokenizer/log sizes):

```json
{
  "run_id": "a762c746d26e485494994ad7dde97fe5",
  "status": "pruned",
  "evaluation_complete": true,
  "resume_supported": false,
  "best_epoch": 2,
  "bytes_before": 2088676,
  "bytes_after": 529526
}
{
  "run_id": "90a8cf3f9ede4a7e876d0128751bc58b",
  "status": "pruned",
  "evaluation_complete": true,
  "resume_supported": false,
  "best_epoch": 2,
  "bytes_before": 1907652,
  "bytes_after": 499044
}
```

Exactly one `.pt` file remains per smoke run. Compact-best standalone evaluation reproduced primary BLEU/chrF and retained run ID, completed training status/time and reference identity. No optimizer/resume state remains; unit tests also reject a mismatched last.pt run ID before deletion. These small-byte savings demonstrate the cleanup path, not a prediction of full-model storage.

The four-way comparison CLI with `--include-cpu --require-four-models` generated Markdown+CSV from fast/large smoke plus the existing local random Marian zero-shot/fine-tuned fixture. CPU mode explicitly permits its small beam width; production strict mode requires full beam4, greedy and schema v2 identities. No real Helsinki GPU artifact is synthesized below:

| Run | Model | Status | Stage | Decode | Beam | BLEU | chrF | Train s | Eval s | Sources | Identity |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| /workspace/marian-final-check/results/finetuned | marian_cpu_fixture | unreported | fine_tuned | beam | 2 | 0.00 | 3.12 | 24.6 | 0.2 | 40 | verified |
| /workspace/marian-final-check/results/finetuned | marian_cpu_fixture | unreported | fine_tuned | greedy | 1 | 0.00 | 3.12 | 24.6 | 0.2 | 40 | verified |
| /workspace/marian-final-check/results/zero_shot | marian_cpu_fixture | unreported | zero_shot | beam | 2 | 0.00 | 3.12 | 0.0 | 0.3 | 40 | verified |
| /workspace/marian-final-check/results/zero_shot | marian_cpu_fixture | unreported | zero_shot | greedy | 1 | 0.00 | 3.12 | 0.0 | 0.3 | 40 | verified |
| results/fast_v4_cpu | scratch_fast_v4_cpu | completed | trained | beam | 4 | 0.00 | 0.00 | 0.2 | 0.3 | 40 | verified |
| results/fast_v4_cpu | scratch_fast_v4_cpu | completed | trained | greedy | 1 | 0.00 | 1.24 | 0.2 | 0.3 | 40 | verified |
| results/stable_large_cpu_v4 | scratch_large_stable_v4_cpu | completed | trained | beam | 4 | 0.00 | 0.00 | 0.3 | 0.1 | 40 | verified |
| results/stable_large_cpu_v4 | scratch_large_stable_v4_cpu | completed | trained | greedy | 1 | 0.00 | 0.00 | 0.3 | 0.1 | 40 | verified |
| results/stable_large_cpu_v4 | scratch_large_stable_v4_cpu | completed | trained | beam/no-repeat-3 | 4 | 0.00 | 0.00 | 0.3 | 0.1 | 40 | verified |

Times are per-run totals repeated per policy; do not sum rows. Full-test scores only. Legacy files lack exact reference-set verification. Helsinki may have Tatoeba pretraining contamination. Different tokenizers/caps/decoding policies are not controlled ablations.


`pip check` reports no broken requirements. Notebook JSON/schema, empty outputs, per-cell descriptions and live streaming checks passed; tracked raw data is unchanged and whitespace checks pass. Full final suite: **34 passed**. New GPU convergence, actual FP16 traces/peak memory and Drive timings remain for the user.
