# User-reported fast Transformer GPU run

> Historical experiment record. Final measured GPU results and publication instructions are in the root README; old timing estimates are superseded.


Source: results supplied in this conversation after the user ran `configs/transformer_colab_fast.yaml` on Colab T4. Raw metrics/log/sample artifacts were not attached, so these numbers have not been independently revalidated and are not emitted as synthetic `metrics.json`.

- 12 epochs, 447 seconds, ~36 seconds/epoch, no early stopping.
- ~85k target tokens/sec; peak memory 1.67 / 15 GB.
- Full test greedy: multi-reference BLEU 54.53 / chrF 67.93, `nrefs:var`.
- 1,024-source beam-3 subset: BLEU 57.11 / chrF 70.38.
- Greedy on that same subset: BLEU 56.61 / chrF 69.77.
- Validation loss decreased 3.77 → 2.06 through the last epoch; train/val gap was small.
- Errors: rare-word mistranslation, omitted tail clauses, missing subjects/repetition, case inflection.

This motivates a larger scratch run while also comparing Helsinki. The small gap/falling validation loss suggests useful optimization/capacity headroom; it does not establish a single cause for every error. Keep full-test and subset scores separate. New result files record exact held-out fingerprints; old files without them remain labeled legacy/unverified by the comparison script.
