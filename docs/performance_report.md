> Historical pre-stability evidence. The user subsequently reported large-model collapse; current diagnosis, safeguards and CPU checks are in [stability_diagnosis.md](stability_diagnosis.md).

> Historical experiment record. Final measured GPU results and publication instructions are in the root README; old timing estimates are superseded.


# Fast Colab training: measured costs and a calibrated time budget

## Goal and scope

Target **2–3 T4 hours for training + final evaluation**. This is not a measured T4 guarantee. The user observed over 25 minutes without completing the first original base epoch, with 3.6/15 GB allocated. Allocation is not utilization; we cannot diagnose AMP or GPU compute occupancy from that number alone. Startup now prints requested/enabled AMP, device, loader settings and batch counts; the first train AND validation batch verifies FP16 logits, and every epoch records peak CUDA allocated memory, time, throughput and scaler-skipped updates.

The actual checked-in corpus is **363,386 raw pairs**, not 500k. Cleaning retains 363,385; train/val/test have 327,013 / 18,248 / 18,124 pairs. No data is silently dropped to meet the budget.

## Changes and measured data

Fast defaults: 256 dimensions, 3 encoder + 3 decoder layers, 4 heads, FFN 1024, joint 8k BPE, max length 24, 12 epochs, patience 3, Noam warmup 800 steps. Model has **9,633,600 parameters**. Base YAML is retained for historical comparisons; the notebook now uses `configs/transformer_colab_fast.yaml`.

SentencePiece was fitted on **all train data only**, then uncapped token lengths measured on that same split (including BOS/EOS):

| Language | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: |
| English | 10 | 16 | 20 | 72 |
| Russian | 10 | 16 | 21 | 86 |

Pair-max p99 = 22. Cap 24 rounds up to a multiple of eight; **1,595 / 327,013 pairs (0.488%)** have at least one side truncated. EOS is preserved. This trades a small long-sentence tail for speed; do not compare cap-24 metrics with cap-128 results as if policies were identical.

Train batching: shuffled 8,192-example pools sorted by length, then batches bounded by **B × (max source length + max target length) ≤ 16,384 padded tokens**, with at most 1,024 sentences. No rows or remainders are dropped. Batch order changes deterministically per epoch. The entire source+target budget includes PAD/BOS/EOS. This is not a per-side 16k budget.

| Policy (same 8k tokenizer and cap 24) | Steps/epoch | Padded tokens/epoch | Padding |
| --- | ---: | ---: | ---: |
| Random 48 sentences | 6,813 | 12,926,469 | 47.9% |
| Length buckets / token budget | 479 | 7,547,463 | 10.8% |

These are epoch-zero measurements; other shuffles can change the counts. Train has **2,993,937 non-PAD target prediction tokens/epoch** after the cap. “Target tokens/sec” excludes BOS, includes EOS and excludes source tokens; “nonpad tokens/sec” counts source+target including BOS/EOS. Keep these definitions when comparing devices.

Two persistent spawned DataLoader workers, prefetch factor 2, pinned memory on CUDA, and nonblocking transfers are configured. CPU token IDs are precomputed once. Spawn avoids forking initialized CPU/CUDA thread pools; Python token lists avoid opening a shared-memory file descriptor per sentence tensor. A 16k-token budget is a starting point, not a promise to fill 15 GB. Inspect `cuda_peak_gb`, tokens/sec and first epoch before raising it. On OOM, reduce `max_tokens` in a copied config and start a new experiment.

## Faster validation and test evaluation

- Validation: full teacher-forced loss with AMP, plus greedy decoding on 128 fixed validation sources. No validation beam search.
- Test: batched, length-sorted greedy on **every** unique test source; beam width 3 on a fixed SHA-256-ordered **1,024-source subset**, with batch size 64. Greedy batch size is 128.
- Sentences and beams execute together on the device. Encoder memory is reused per decode call; the output projection only processes the newest decoder position. Decoder prefixes are still recomputed (no KV cache); the 24-token cap limits that cost.
- `test_bleu`/`test_chrf` are full-test GREEDY scores in this config. `beam_subset_*` and `beam_subset_greedy_*` are same-subset comparisons, reported separately. No mixture of greedy and beam output is called a homogeneous beam score. Reference alignment is restored after length sorting.
- Set `--beam-max-sources 1000000` for a deliberate full-beam run, outside the default budget. Set YAML `beam_max_sources: 0` to disable subset beam.

## Actual CPU throughput and conditional T4 estimate

Reproduce:

```bash
python -m src.benchmark --config configs/transformer_colab_fast.yaml \
  --output results/fast_cpu_benchmark/throughput.json
python -m src.train --config configs/transformer_fast_cpu.yaml
```

Throughput measurement uses the **same 256×3 architecture and full train-fitted 8,000-piece vocabulary**. Only CPU batch limits and the number of observed batches are reduced: max 2,048 padded tokens / 64 sentences, two CPU threads; 2 warmup updates then 12 timed updates from a 4,096-pair subset. It includes forward/backward, Adam, clipping and Noam, but excludes tokenizer fit and batch materialization. FP32, Python 3.12.14, PyTorch 2.6.0+cpu.

Measured: **1,336.7 target tokens/sec**, **3,040.3 source+target non-PAD tokens/sec**, 6,495 target tokens in 4.859 s. CPU performance does **not** determine a T4 speedup ratio; Tensor Cores, batching, transfer and memory behavior differ. We therefore expose assumptions rather than claiming a GPU measurement.

With 12 × 2,993,937 target tokens, assume 20% overhead for validation/checkpoints and 15 minutes for final evaluation:

`total_hours = 12 × target_tokens_per_epoch / assumed_T4_target_tokens_per_second × 1.20 / 3600 + 0.25`

| Assumed T4 rate vs measured CPU | Target tokens/sec | Conditional total |
| --- | ---: | ---: |
| 3.5× (assumption) | 4,678 | 2.81 h |
| 5× (assumption) | 6,683 | 2.04 h |
| 10× (assumption) | 13,367 | 1.15 h |

Thus the **2–3 hour target requires approximately 4,355–6,843 target tokens/sec on the T4** under these overhead assumptions. A 3.5–5× CPU-rate scenario gives approximately **2.81–2.04 hours**; these multipliers are assumptions, not measured T4 results. The estimate replaces the old 12–48-hour planning range for the fast config only. For comparison, no cross-device extrapolation or small CPU smoke score is translation-quality evidence.

## One-epoch timing and live output

The Colab notebook uses a new `scratch_fast_v2_seed42` experiment. Old base-model checkpoints cannot resume into different dimensions/vocabulary/cap/batching. For a new fast run:

```bash
python -u -m src.train --config configs/transformer_colab_fast.yaml \
  --stop-after-epochs 1 --skip-test
# Same YAML still plans 12 epochs, so early stopping/Noam/RNG stay intact:
python -u -m src.train --config configs/transformer_colab_fast.yaml \
  --resume checkpoints/transformer_colab_fast/last.pt
```

The notebook streams both stdout and stderr using `Popen`, line-by-line flushes and unbuffered Python. tqdm shows **step/total, ETA, loss and tokens/sec**; epoch summaries show train/validation time, checkpoint time, remaining training ETA and projected train+test time. After epoch one a separate cell prints `runtime_estimate.json`.

That estimate uses measured first/recent epoch wall time (train + validation loss + greedy validation), observed checkpoint time, and validation greedy sources/sec. Final test cost extrapolates to all test sources plus the beam subset, with a **width× greedy cost assumption** for beam. It excludes one-time preprocessing, warns if the target is exceeded, and does not silently drop epochs/data. Actual generated lengths, checkpoints and final decode timing may differ. Early stopping can shorten the run. The last save's checkpoint-time field can be zero in a portable checkpoint because it cannot know its own write duration; the current CSV/estimate records that duration.

## CPU correctness validation

The reduced fast-pipeline config uses 256 train pairs, 32 validation pairs and 40 test pairs, token buckets, two workers, two epochs, full greedy and a separate 16-source beam subset. The notebook is valid nbformat v4 JSON with outputs cleared. Tests cover exact resume with dropout and spawned workers, batched versus individual decoding, beam subset labeling, mask/shape invariants, tokenizer and one training update. See the measured validation appendix below for final logs.

## Remaining checks for the user GPU run

Actual T4 throughput, FP16 dtype logs, peak memory, first-epoch projection, full GPU convergence, and Colab/Drive execution still need observation. No GPU training was run in the cloud CPU environment. Send `metrics.json`, `train_log.csv`, `samples.txt`, `runtime_estimate.json`, config, commit, GPU model and any warnings after running.

## Measured validation appendix

```text
.................                                                        [100%]
17 passed in 13.62s
```

The streaming test runs the notebook's actual helper with a child that prints to stdout, waits, then prints to stderr; the first line arrives before child exit. The subprocess return code is propagated.

CPU fast-pipeline invocation:

```bash
python -m src.train --config configs/transformer_fast_cpu.yaml \
  --checkpoint-dir checkpoints/fast_cpu_v2 --results-dir results/fast_cpu_v2
python -m src.evaluate --checkpoint checkpoints/fast_cpu_v2/last.pt \
  --results-dir results/fast_cpu_v2_recheck
```

Actual CSV (small model, not the full-architecture throughput measurement above):

```csv
epoch,train_loss,val_loss,lr,epoch_seconds,checkpoint_seconds,train_seconds,val_seconds,target_tokens_per_second,nonpad_tokens_per_second,padding_fraction,val_greedy_bleu,val_greedy_chrf,val_greedy_seconds,val_greedy_sources,cuda_peak_gb,amp_enabled,logits_dtype,skipped_steps
1,5.034047905042364,4.713074048913043,0.030383494504109466,3.311184284000774,0.014932897000107914,1.7766788089993497,1.4731751569997869,2723.6211607236946,5578.385890459296,0.0788177339901478,0.0,1.201923076923077,0.028433735000362503,16,0.0,False,torch.float32,0
2,4.713254288076048,4.4778035835597825,0.037688918072220454,0.16359314100009215,0.014147991999379883,0.12168477999966854,0.014274629000283312,39766.6824068974,81448.14824028936,0.0788177339901478,0.8505431128716193,1.9668737060041408,0.026993050999408297,16,0.0,False,torch.float32,0
```

Actual smoke metrics (not quality claims):

```json
{
  "test_decoding": "greedy",
  "test_sources": 40,
  "test_bleu": 0.41450283043775876,
  "test_chrf": 1.8213001925564256,
  "beam_subset_sources": 16,
  "beam_subset_bleu": 0.0,
  "beam_subset_chrf": 4.631985528118482,
  "training_seconds": 3.474777425000866,
  "evaluation_seconds": 0.16094029000032606
}
```

Both epochs, full greedy, subset beam, 30 saved examples, metrics/log/ETA files and standalone evaluation completed. Standalone full/subset scores matched exactly. `pip check` reported no broken requirements; notebook validity/empty outputs and repository whitespace checks passed. CPU AMP is correctly disabled and logs float32; GPU fp16 still needs the first T4 batch check.
