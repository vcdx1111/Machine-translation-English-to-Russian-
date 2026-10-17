# Final publication validation

Validation was run on CPU with Python 3.12, torch 2.6.0+cpu and the pinned base/Helsinki requirements. No additional full-data model training or GPU training was performed.

## Executed checks

- `python -m pytest -q`: **39 passed**, 14 third-party matplotlib/pyparsing deprecation warnings, **118.78 seconds**. No skipped or deselected tests in the final run.
- `python -m pip check`: no broken requirements.
- All three notebooks pass nbformat JSON/schema validation, with every code-cell output empty and execution count null.
- `python scripts/build_full_notebook.py --check`: the standalone notebook matches its embedded pipeline/config sources.
- `git diff --check`: passed.
- Git ignore probes confirm processed data, checkpoint directories, `.pt`, `.safetensors`, `.env` and Python caches are ignored; `data/raw/rus.txt` is allowed.

## Standalone notebook execution

The full CPU smoke executed **17 code cells in 86.33 seconds**, in a new working directory with a Python import hook that rejects all `src`/`scripts` imports. The actual raw corpus was audited and split first. Tiny training limits were then applied: 48 train / 8 validation / 40 test pairs, seed 42. Scratch dimensions were reduced to d32, 1 layer, FF64 and 256-token BPE; fast retained Post-LN/untied behavior and large retained Pre-LN/scaled initialization/tied weights. Each ran one CPU epoch. Marian used a local **random d16/1-layer fixture**, not pretrained Helsinki.

| Stage | Test sources | Status | Artifacts |
| --- | ---: | --- | --- |
| fast | 40 | completed | metrics.json, samples.txt (30), train_log.csv |
| large | 40 | completed | metrics.json, samples.txt (30), train_log.csv |
| zero_shot | 40 | completed | metrics.json, samples.txt (30), train_log.csv |
| finetuned | 40 | completed | metrics.json, samples.txt (30), train_log.csv |

All four fixture runs used test hash `0173433289587b69fda744a3aad3acc54164dbdf10f667e26b6dcfd5394eab0a`. This tiny test identity is deliberately different from the full-test hash; CPU scores do not enter the final GPU table.

The smoke also verified full-subset beam4 plus greedy, large no-repeat-3 evaluation, matching comparison identities, generated Markdown/CSV examples/comparison and PNG score/loss plots. It found exactly two scratch `.pt` files, both named `best.pt`, one Marian best safetensors file and no `checkpoint-N` directories. The zero-shot fixture has a CSV header only; the fine-tuned fixture has its actual epoch log. Separate regression coverage confirms legacy Helsinki completion without a CSV and preserves explicit failure statuses.

The production full test was independently recomputed from the checked-in raw data: **18,124 pairs / 12,556 exact English sources**, SHA-256 `bc6e2388f0938c3332e747f174c98d16530f69b4641f0cb6ddb3c074e8f39f30`.

Reproduce the standalone check with a new output directory:

```bash
python scripts/validate_full_notebook.py --output /tmp/en-ru-notebook-smoke
```

The command writes `execution.log` and `validation.json` alongside the ignored/out-of-repository smoke artifacts. Each new invocation requires a fresh directory to avoid accidentally reusing weights.

## Public-history audit

`python scripts/audit_publication.py` scans all blobs reachable through local Git refs, including historical versions, for checkpoint/environment-file paths, unexpected blobs over 10 MiB and common GitHub/Hugging Face/AWS/Google/OpenAI token, private-key and credential-URL formats. No matching secret formats or checkpoint/environment artifacts were found. The only large blob is the intentionally tracked raw corpus: **60,062,354 bytes** (about 57.3 MiB). It is below GitHub’s 100 MiB per-file limit. The scan is rerun on the final committed tree before push.

Regex scanning cannot prove the absence of every possible credential, and does not inspect unreachable server objects or ignored local files. No existing Git history was rewritten. Temporary CPU artifacts and model weights remain outside tracked files.

## What was not re-verified

- The final T4 quality scores and epoch endpoints are user-reported measurements. Original GPU metrics/logs/samples are in Drive and are not accessible in this cloud workspace.
- CUDA/FP16 behavior and full-size runtime of the new standalone notebook were not executed here. Its model/optimization/evaluation code is embedded from the reviewed modular sources; configs match, with zero notebook DataLoader workers for interactive-cell compatibility.
- The CPU Marian fixture checks Trainer/generation/artifact plumbing, not Helsinki pretrained quality. Real Hugging Face weights were not downloaded or evaluated in this environment.
- Exact fast/zero-shot/fine-tuned outputs for the requested error cases were not supplied. README marks them unavailable; join saved samples or use the optional checkpoint-inference cell without training.
- The actual large v4 loss curve awaits its real Drive CSV; no synthetic epoch values or missing Helsinki CSV logs were manufactured.

## Remaining publication artifacts

Run the README reporting commands in Colab with Drive mounted, commit the two figure PNGs and provenance file, and copy exact four-way examples into the README. If the 30 saved samples omit the requested sentences, enable only `RUN_EXAMPLE_INFERENCE` in the standalone notebook, using the original saved checkpoints and pinned base revision. Keep every training flag False. Clear all notebook outputs before committing.
