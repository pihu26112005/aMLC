# aMLC Business Entity Resolution

An end-to-end, large-scale entity-resolution pipeline for matching Source 1 businesses with zero or more records from Sources 2 and 3. The solution combines multi-pass Sorted Neighbourhood candidate generation, engineered string/address features, hard-negative mining, and a LightGBM pair classifier.

The best public leaderboard score obtained by this implementation was **0.905 macro F0.5**.

## Approach at a glance

```text
Raw S1/S2/S3 records
        |
        v
Unicode/transliteration-aware normalization
        |
        v
Eight country-partitioned Sorted Neighbourhood passes
        |
        v
Union and deduplicate candidate pairs
        |
        v
33 retrieval, name, address and numeric features
        |
        v
LightGBM + weighted hard-negative mining
        |
        v
Thresholding + target-side uniqueness post-processing
        |
        v
matching_results.tsv + candidate_pairs.tsv
```

This is not just a sorting or binary-search solution. Sorting is used to reduce the otherwise quadratic search space; a supervised LightGBM classifier makes the final pair-level match decision.

## Repository contents

```text
.
├── aMLC_sorted_neighborhood_approach_enhanced.ipynb
├── README.md
├── REPORT.md
├── models/
│   ├── enhanced_lightgbm.txt
│   ├── enhanced_validation_metrics.json
│   ├── enhanced_threshold_results.csv
│   └── enhanced_feature_importance.csv
└── outputs/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

`REPORT.md` contains the complete methodology, validation results, design decisions, limitations, and improvement ideas.

## Main techniques

- NFKC Unicode normalization, transliteration, case folding, punctuation cleanup, and whitespace normalization.
- Legal-suffix removal and derived name/address signatures.
- Eight complementary Sorted Neighbourhood passes with country isolation.
- Resumable Parquet shards and persistent Google Drive checkpoints.
- 33 pair features covering candidate provenance, names, addresses, numbers, postal codes, conflicts, and missingness.
- LightGBM trained with bounded random negatives and up to two million mined hard negatives.
- Separate S2/S3 threshold search followed by target-side uniqueness and confidence-margin filtering.
- Country-aware test calibration for France, which was absent from training data.

## Key results

| Stage | Result |
|---|---:|
| Initial simpler pipeline | 0.844 public score |
| Enhanced features and classifier | 0.875 public score |
| Target-side uniqueness post-processing | 0.900 public score |
| Country-aware France calibration | **0.905 public score** |
| Validation candidate-link recall | 0.9005 |
| Validation candidate oracle macro F0.5 | 0.9608 |
| Full validation score after uniqueness | approximately 0.9136 |

- https://amazonmlchallengeexplorer.vercel.app/organizations/indian-institute-of-technology-iit-mandi
