# Generated results

Current fast/large and Helsinki stage directories use schema v2: `metrics.json`, `train_log.csv`, `samples.txt`, exact held-out identity/signatures/config and separate full-test greedy/beam4 scores. Zero-shot has training time 0 and CSV header only. Older fast subset-beam scores are historical fields and must not be substituted for full-test results.

Scratch v4 also writes step logs (unscaled gradient norm, clip bound, proposed/applied LR, scaler, skip), run status, ETA and divergence diagnostics. Large's trigram-suppressed scores/samples are separate; primary beam scores always have suppression off. Each scratch experiment has a run ID for checkpoint cleanup safety.

`python scripts/compare_results.py <fast> <large> <helsinki root> --require-four-models --output results/comparison_v4` emits Markdown/CSV and requires matching schema v2 identities with full beam4+greedy. CPU smoke files are excluded unless explicitly requested; their scores are plumbing checks only. The strict flag's beam requirement is relaxed for local CPU fixtures. Times repeat the run total per policy and must not be summed across rows. Helsinki may have OPUS/Tatoeba pretraining overlap and unsupported extra tails despite high BLEU.

After completed evaluation, `python -m src.checkpoints ...` keeps one compact inference-only best.pt, removes last/resume duplicates, preserves all result/tokenizer files, and records cleanup space. Cleanup refuses running, calibrated, diverged or mismatched runs. Compact best cannot resume training. Weights and generated result files stay ignored; curated user GPU results can be committed separately after artifact review. Do not synthesize metrics.json from chat summaries.
