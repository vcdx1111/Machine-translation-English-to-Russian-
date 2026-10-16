# Generated results

Each run writes `metrics.json` (test BLEU/chrF, metric signatures, timing, raw-data hash and full config), `train_log.csv`, and `samples.txt` (up to 30 unique sources and all available references). The default full and CPU test splits both contain at least 30 unique sources. Small custom datasets may contain fewer.

Run outputs and weights are ignored. CPU smoke scores prove execution only and must not populate the portfolio's model quality table. Commit curated full GPU results separately after review. Training time excludes preprocessing and test decoding; evaluation time is recorded separately. SentencePiece models are bundled in trusted checkpoints.
