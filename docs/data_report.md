# Data report

Generated from the checked-in `data/raw/rus.txt`; no download is required.

Each row has English, Russian, and Tatoeba attribution, separated by tabs. Attribution is retained in processed rows. Source: Kaggle “English-Russian dictionary for machine translate”, originally Tatoeba (CC-BY 2.0); the exact Kaggle listing URL was not supplied. Preserve the original attribution and review the dataset license before redistribution; rows identify CC-BY 2.0 (France).

## Cleaning and leakage prevention

NFC Unicode normalization and whitespace collapse; preserve case, punctuation, and Russian stress accents. Remove empty text, missing expected Latin/Cyrillic scripts, control characters, duplicate normalized pairs, and pairs over 300 characters or 60 whitespace words on either side. These are conservative heuristics, not semantic quality labels. No automatic language detector or manual quality certification is claimed.

Group by casefolded English alphanumeric characters (punctuation/whitespace removed). Shuffle sorted groups with seed 42; allocate 90/5/5 percent of groups. All translations and simple punctuation/case variants stay together. Paraphrases can still overlap; sentence-level splitting does not guarantee topic isolation. Fit SentencePiece on train only.

## Measured statistics

```json
{
  "sha256": "d5105634aef975c9ddc9bfa0862cc8f882cd29fa7a97bde2a5e3af18e5ad0504",
  "seed": 42,
  "raw_lines": 363386,
  "column_counts": {
    "3": 363386
  },
  "malformed": 0,
  "missing": {
    "en": 0,
    "ru": 0,
    "attribution": 0
  },
  "noise_observations": {
    "non_nfc_rows": 0,
    "whitespace_normalized_rows": 72,
    "identical_normalized_text": 0,
    "markup_like_rows": 0,
    "replacement_character_rows": 0
  },
  "raw_duplicate_pairs": 0,
  "sources_with_multiple_translations": 74440,
  "lengths": {
    "en": {
      "characters": {
        "min": 3.0,
        "p50": 28.0,
        "p90": 43.0,
        "p95": 49.0,
        "p99": 64.0,
        "max": 537.0
      },
      "words": {
        "min": 1.0,
        "p50": 6.0,
        "p90": 9.0,
        "p95": 10.0,
        "p99": 13.0,
        "max": 101.0
      }
    },
    "ru": {
      "characters": {
        "min": 4.0,
        "p50": 28.0,
        "p90": 44.0,
        "p95": 51.0,
        "p99": 67.0,
        "max": 475.0
      },
      "words": {
        "min": 1.0,
        "p50": 5.0,
        "p90": 8.0,
        "p95": 9.0,
        "p99": 12.0,
        "max": 80.0
      }
    }
  },
  "removed": {
    "length": 1
  },
  "retained": 363385,
  "split_rows": {
    "train": 327013,
    "val": 18248,
    "test": 18124
  },
  "split_groups": {
    "train": 225753,
    "val": 12542,
    "test": 12542
  },
  "multitranslation_examples": [
    {
      "en": "Go.",
      "ru": [
        "Иди.",
        "Идите.",
        "Марш!"
      ]
    },
    {
      "en": "Hi.",
      "ru": [
        "Здоро́во!",
        "Здравствуйте.",
        "Здрасте.",
        "Привет!",
        "Хай."
      ]
    },
    {
      "en": "Run!",
      "ru": [
        "Беги!",
        "Бегите!"
      ]
    },
    {
      "en": "Run.",
      "ru": [
        "Беги!",
        "Бегите!"
      ]
    },
    {
      "en": "Wow!",
      "ru": [
        "Вах!",
        "Вот это да!",
        "Здорово!",
        "Круто!",
        "Ого!"
      ]
    }
  ]
}
```
