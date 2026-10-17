# Corpus provenance and license

`raw/rus.txt` was supplied from the Kaggle dataset **English-Russian dictionary for machine translate**, originating from [Tatoeba](https://tatoeba.org/). The exact Kaggle listing URL was not supplied.

The corpus is **[Creative Commons Attribution 2.0](https://creativecommons.org/licenses/by/2.0/)**; individual row credits identify the [France port](https://creativecommons.org/licenses/by/2.0/fr/). Each TSV row retains its original English, Russian and contributor attribution. Preserve those credits in redistribution. The repository's MIT license applies to code and does not relicense the corpus.

The raw file is unmodified: 363,386 rows, three columns, SHA-256 `d5105634aef975c9ddc9bfa0862cc8f882cd29fa7a97bde2a5e3af18e5ad0504`. Cleaning is reproducible and writes only ignored `data/processed/` files. See `docs/data_report.md` and the root README for rules and split identities.
