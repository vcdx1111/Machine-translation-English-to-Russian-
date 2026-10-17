# Historical configurations

`transformer_base_initial.yaml` records the initial unoptimized baseline, before bucketing/short-sequence optimization. `transformer_large_v3.yaml` preserves the failed large Post-LN experiment from commit `ab8595c` (warmup 500, factor 0.5, 32,768 tokens). It relies on model defaults for Post-LN, untied output and legacy Xavier initialization. Running it on current code adds current diagnostics; it does not restore old implementation bugs.

Final configurations are `../transformer_colab_fast.yaml` and `../transformer_colab_large.yaml`. The historical configurations are not recommended training targets. Relative paths inside all YAML files are interpreted from the repository root.
