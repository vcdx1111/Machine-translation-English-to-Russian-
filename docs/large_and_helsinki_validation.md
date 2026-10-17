> Historical pre-stability evidence. The user subsequently reported large-model collapse; current diagnosis, safeguards and CPU checks are in [stability_diagnosis.md](stability_diagnosis.md).

> Historical experiment record. Final measured GPU results and publication instructions are in the root README; old timing estimates are superseded.


# Larger scratch + Helsinki CPU validation

No GPU training was performed. The user's fast GPU run is separately reported in `user_reported_fast_run.md`. The new large GPU run and actual pretrained Helsinki zero-shot/fine-tuning remain for Colab.

## Dependency/API environment

Python 3.12.14, torch 2.6.0+cpu, Transformers 4.49.0, Accelerate 1.4.0; dependencies pinned in `requirements-helsinki.txt`. Official CPU torch/torchvision wheels preserve TLS verification. Real Helsinki model config access at `huggingface.co` failed with proxy CONNECT 403; no pretrained weights or quality were validated here. No credentials were requested or bypassed.

## Pytest

```text
........................                                                 [100%]
24 passed in 22.70s
```

Includes prior masks/shapes/tokenizer/update/beam/streaming tests plus ngram masking on/off, all-source large evaluation policies, exact test identity/comparison mismatch checks, both notebook schemas/empty outputs, Marian CPU label shift, underlying Accelerate loader spawn selection, zero-shot-before-training artifacts, and **bitwise identical full-model weights after uninterrupted vs resumed two-epoch tiny Marian training**. The latter includes optimizer/linear schedule/RNG/early-stopping resume and validates every state-dict tensor, including tied aliases.

## Scratch large pipeline smoke

```bash
python -m src.train --config configs/transformer_large_cpu.yaml
```

96 train pairs / 16 validation / 40 test, reduced model, two epochs. Actual log:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds,checkpoint_seconds,train_seconds,val_seconds,target_tokens_per_second,nonpad_tokens_per_second,padding_fraction,val_greedy_bleu,val_greedy_chrf,val_greedy_seconds,val_greedy_sources,cuda_peak_gb,amp_enabled,logits_dtype,skipped_steps
1,5.292004139310301,4.773346729767629,0.019531250000000003,0.1967592799992417,0.021461345997522585,0.13356756500070333,0.013850791998265777,14681.708092751964,29445.771508818703,0.1354143767861068,0.0,0.0,0.0057008090007002465,8,0.0,False,torch.float32,0
2,4.823718765935747,4.872372749524239,0.027950849718747374,0.21713374899991322,0.008431981998000992,0.1444504609971773,0.006470278000051621,13575.588381392012,27227.327437029468,0.1354143767861068,0.0,0.0,0.06479648499953328,8,0.0,False,torch.float32,0
```

All 40 test sources evaluated with greedy, beam 4 and beam 4/no-repeat-3. Primary and ablation samples saved separately, 30 examples each. Metrics/log/ETA files saved. Actual smoke metrics:

```json
{
  "test_sources": 40,
  "test_decoding": "beam",
  "test_beam_size": 4,
  "test_bleu": 0.0,
  "test_chrf": 0.0,
  "test_greedy_bleu": 0.0,
  "test_greedy_chrf": 0.0,
  "training_seconds": 0.4138930289991549,
  "evaluation_seconds": 0.11838651000289246,
  "test_split_sha256": "0173433289587b69fda744a3aad3acc54164dbdf10f667e26b6dcfd5394eab0a",
  "repetition_ablation": {
    "no_repeat_ngram_size": 3,
    "beam_size": 4,
    "bleu": 0.0,
    "chrf": 0.0,
    "bleu_signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1",
    "chrf_signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.5.1",
    "sources": 40
  }
}
```

## Local Marian Trainer pipeline smoke

```bash
python -m scripts.make_marian_fixture --output /workspace/marian-cpu-fixture
python -m src.helsinki --config /workspace/marian-cpu-fixture/fixture.yaml \
  --stop-after-epochs 1 --skip-test
python -m src.helsinki --config /workspace/marian-cpu-fixture/fixture.yaml --resume latest
```

The fixture is a **random, 16-dimensional, one-layer Marian** and train-fitted local SentencePiece tokenizer, not Helsinki pretrained weights. It uses 48 train / 8 validation / 40 test pairs from the real grouped split. Zero-shot (random fixture baseline) wrote metrics/header-only CSV/30 samples before updates; epoch one saved Trainer checkpoint-6; resume completed epoch two and wrote best-model evaluation, final model/tokenizer, manifest, CSV, metrics and samples. Actual observed epoch log:

```csv
epoch,train_loss,val_loss,lr,epoch_seconds,global_step,cuda_peak_gb
1.0,5.546833333333333,5.525830268859863,1.6363636363636363e-05,0.19020882999757305,6.0,0.0
2.0,5.545066666666667,5.525271415710449,0.0,0.27750487199955387,12,0.0
```

The fixture and scratch smoke used the **same 40-source reference identity**:

`0173433289587b69fda744a3aad3acc54164dbdf10f667e26b6dcfd5394eab0a`

The comparison script rejects a changed identity and excludes these files by default because their purpose is CPU smoke. Separate CPU execution also exercises two worker processes; see the final verification note below.

## Reproduction and limits

The full training-only 16k BPE profile (`large_train_profile.json`) measured 279 batches/epoch, 17.9% padding, 500-step warmup ≈1.79 epochs and one training pair over the 64-token cap. Profiling did not train the large model.

New model T4 memory fit, actual pretrained download/revision propagation, GPU AMP/convergence and Colab/Drive duration remain unverified. Large planning range is 60–150 minutes; Helsinki targets 1–2 hours. These estimates are not measured on the new workflows. Both notebooks measure a real first epoch and stream child stdout/stderr; use those measurements instead of presenting estimates as results. Marian reload messages about omitted tied aliases and Transformers' internal tokenizer deprecation do not change the bitwise CPU resume check.

## Final worker and comparison verification

The two-worker resumed CLI used spawned train/evaluation workers on Accelerate's underlying DataLoader, with a CPU-only label-shifting collator. It completed epoch two and final generation. All final state-dict tensors were bitwise identical to the zero-worker uninterrupted/reference result for the same local base/data. The relevant second-epoch wall time included process startup (~7.6 s in this tiny fixture), so it is not a GPU throughput estimate.

The CLI comparison successfully created `results/ab_cpu_comparison.md` and `.csv` using the scratch smoke plus Marian zero-shot/fine-tuned smoke. The following is CPU plumbing evidence only; no pretrained or scratch model quality is claimed:

| Run | Model | Stage | Decode | Beam | BLEU | chrF | Train s | Eval s | Sources | Identity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| /workspace/marian-final-check/results/finetuned | marian_cpu_fixture | fine_tuned | beam | 2 | 0.00 | 3.12 | 24.6 | 0.2 | 40 | verified |
| /workspace/marian-final-check/results/finetuned | marian_cpu_fixture | fine_tuned | greedy | 1 | 0.00 | 3.12 | 24.6 | 0.2 | 40 | verified |
| /workspace/marian-final-check/results/zero_shot | marian_cpu_fixture | zero_shot | beam | 2 | 0.00 | 3.12 | 0.0 | 0.3 | 40 | verified |
| /workspace/marian-final-check/results/zero_shot | marian_cpu_fixture | zero_shot | greedy | 1 | 0.00 | 3.12 | 0.0 | 0.3 | 40 | verified |
| results/large_cpu | scratch_transformer | trained | beam | 4 | 0.00 | 0.00 | 0.4 | 0.1 | 40 | verified |
| results/large_cpu | scratch_transformer | trained | greedy | 1 | 0.00 | 0.00 | 0.4 | 0.1 | 40 | verified |
| results/large_cpu | scratch_transformer | trained | beam/no-repeat-3 | 4 | 0.00 | 0.00 | 0.4 | 0.1 | 40 | verified |

Times are per-run totals repeated per policy; do not sum rows. Full-test scores only. Legacy files lack exact reference-set verification. Helsinki may have Tatoeba pretraining contamination. Different tokenizers/caps/decoding policies are not controlled ablations.

The comparison formatting update was checked with the existing mismatch/CSV/Markdown test:

```text
.                                                                        [100%]
1 passed in 1.78s
```

A final fresh two-worker run at `/workspace/marian-final-check/` repeated zero-shot → epoch-one stop → `latest` resume → best-model evaluation with the final code. Both stages explicitly record `model_family=marian_cpu_fixture`, `pretrained_model_validated=false`, 40 test sources and 30 samples. Their exact held-out hashes match the scratch smoke. Its comparison files above were regenerated from these artifacts. These are random-model API tests, not a Helsinki score report.
