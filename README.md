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

The oracle result shows that candidate generation, rather than only classifier thresholding, became the principal remaining bottleneck.

## Running the notebook

The notebook was designed for Google Colab and contains numbered blocks. Run the blocks sequentially for a fresh execution. It checkpoints normalized data, candidate shards, feature shards, validation predictions, model artifacts, and test selections to Google Drive, allowing recovery after a runtime reset.

For restart recovery, follow the recovery block included after test candidate generation, then continue from the first incomplete stage. Do not run a later block in a new runtime until its imports, configuration, helper functions, and required local files have been restored.

Hardware used:

- GPU for large-scale Sorted Neighbourhood candidate generation.
- CPU/DuckDB for bounded-memory joins, feature construction, validation, post-processing, and TSV generation.
- LightGBM training is CPU-compatible.

## Submission files

- `matching_results.tsv`: one row for every S1 test entity and a comma-separated list of predicted S2/S3 matches.
- `candidate_pairs.tsv`: one row for every S1 test entity and the full candidate list produced by blocking.

Both files are tab-separated and retain empty match lists for predicted singletons.

### GitHub file-size warning

The generated `candidate_pairs.tsv` is approximately 2.78 GB and cannot be committed to ordinary GitHub storage. `matching_results.tsv` is also close to GitHub's 100 MB per-file limit. Store these with Git LFS, attach them to a release where permitted, or provide a cloud-storage link. Do not accidentally commit training/test datasets if the competition rules prohibit redistribution.

## Reproducibility notes

- Random seed: `42`.
- Deterministic hash-based train/validation split.
- Candidate and feature data divided into 32 shards.
- The trained model and its exact feature list are stored together with validation metrics.
- Best public configuration: US/India threshold `0.37`, France threshold `0.85` (with `0.90` tying on the public leaderboard), and target uniqueness margin `0.02`.
- A later France-margin experiment using `0.10` scored lower (`0.903`) and should not be confused with the best configuration.

## Limitations

- Training data contains US and India, while test data also contains France; France calibration therefore relied on public-leaderboard feedback and may overfit the public split.
- Candidate recall is about 90%, so the classifier cannot recover true pairs absent from the candidate set.
- The pipeline is computationally and storage intensive despite bounded-memory sharding.
- Public leaderboard performance may differ from the private leaderboard.

## Responsible use

The solution uses only the supplied records and derived features. External business-entity lookup should not be added where prohibited by the competition rules.
