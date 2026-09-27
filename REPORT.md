# Technical Report: Enhanced Multi-Pass Sorted Neighbourhood Entity Resolution

## 1. Problem definition

The task is to resolve business entities across three independently collected sources. Source 1 is the deduplicated reference source. For every Source 1 entity, the system must return zero, one, or many matching Source 2 and Source 3 entity identifiers.

The output is evaluated using macro F0.5. This metric gives precision twice the importance of recall, so an incorrect merge is more damaging than a missed link. Source 1 singletons are included in the macro average: predicting an empty list for a true singleton is valuable, whereas attaching an unrelated record is penalized.

A direct comparison of every S1 record with every S2/S3 record is infeasible. The solution therefore has two distinct learned-search stages:

1. **Candidate generation:** efficiently retrieve a small but high-recall set of plausible S2/S3 records for each S1 entity.
2. **Pair classification:** estimate the probability that each retrieved pair represents the same real-world business.

## 2. Final system architecture

The final solution is an enhanced multi-pass Sorted Neighbourhood pipeline followed by a supervised LightGBM classifier and conservative graph-style post-processing.

```text
S1, S2 and S3 TSV files
    -> normalized Parquet records and derived signatures
    -> country-partitioned multi-pass Sorted Neighbourhood
    -> candidate union, deduplication and retrieval metadata
    -> pairwise feature engineering
    -> weighted LightGBM training with hard-negative mining
    -> validation threshold optimization
    -> test scoring
    -> target-side uniqueness and confidence-margin filtering
    -> country-aware France calibration
    -> exact leaderboard TSV files
```

Intermediate data is written as compressed Parquet and split into 32 deterministic shards. Every expensive stage is checkpointed to Google Drive so that a Colab runtime interruption does not force a complete restart.

## 3. Data preparation and normalization

### 3.1 Text normalization

Names, addresses, and country values are normalized using:

- Unicode NFKC normalization;
- `unidecode` transliteration to a comparable Latin representation;
- case folding;
- replacement of non-alphanumeric characters with spaces;
- whitespace collapse and trimming.

This reduces differences caused by capitalization, accents, punctuation, and formatting. Transliteration is especially relevant because source systems may represent the same business using different scripts or accent conventions.

### 3.2 Derived signatures

The notebook derives several structured representations from the normalized text:

- `name_core`: business name after trailing legal suffix removal;
- `name_acronym`: initials of meaningful name tokens;
- sorted unique name tokens;
- sorted unique address tokens;
- address numeric signature;
- first number as the probable house number;
- five- or six-digit postal-code signature;
- final three nonnumeric address tokens as the address tail.

Legal suffix removal covers common forms such as `inc`, `corp`, `llc`, `ltd`, `limited`, `pvt`, `gmbh`, `sa`, `sarl`, `bv`, and related variants. Raw normalized values are preserved alongside the derived forms so that suffix removal cannot destroy all evidence available to the classifier.

## 4. Candidate generation

### 4.1 Why Sorted Neighbourhood

Sorted Neighbourhood sorts records by a comparison key and compares records occurring within a bounded rank window. Similar records tend to become close after normalization, reducing an all-pairs comparison to a tractable local-neighbour search.

One sort key cannot handle every corruption pattern. The final pipeline therefore uses eight complementary passes. Each pass combines several signals and has its own radius.

| Pass | Main key components | Radius |
|---|---|---:|
| `name` | normalized name, postal code, house number | 20 |
| `name_tokens` | sorted name tokens, postal code, house number | 20 |
| `address_numbers` | numeric signature, address tail, core name | 12 |
| `address_tokens` | sorted address tokens, core name | 15 |
| `name_core` | suffix-stripped name, postal code, address tail | 20 |
| `name_acronym` | name acronym, postal code, house number | 8 |
| `postal_house` | postal code, house number, core name | 12 |
| `address_tail` | address tail, house number, core name | 12 |

### 4.2 Country partitioning

Records are processed within the same normalized country. This prevents unrelated cross-country comparisons, reduces work, and improves candidate precision. S1, S2, and S3 are combined for sorting, but only S1-to-S2/S3 pairs are emitted.

### 4.3 Multi-pass union and metadata

Candidate pairs from all passes are unioned and deduplicated. The system retains:

- minimum observed rank distance;
- number of passes that retrieved the pair;
- one binary provenance flag for every pass;
- whether the target belongs to S2 or S3.

These values are not only operational metadata; they become classifier features. A pair found by several independent keys is generally more credible than a pair found by only one weak pass.

### 4.4 Bounded-memory implementation

Candidate generation is performed on GPU, while unioning uses DuckDB and Parquet. Pairs are deterministically partitioned into 32 shards using a hash of the S1 identifier. This avoids materializing the complete candidate table in RAM.

The enhanced validation candidate generator achieved:

- candidate-link recall: **0.9005237**;
- at-least-one coverage: **0.9857111**;
- complete-match coverage: **0.7409166**;
- candidate oracle macro F0.5: **0.9607998**.

The oracle score evaluates an ideal classifier that selects every true link present in the generated candidates. Consequently, it establishes an upper bound for the downstream classifier using this candidate set.

## 5. Pairwise feature engineering

The classifier uses 33 numeric features grouped as follows.

### 5.1 Retrieval features

- minimum Sorted Neighbourhood rank distance;
- number of retrieving passes;
- eight pass-provenance flags;
- S3 target indicator.

### 5.2 Name features

- normalized-name exact match;
- core-name exact match;
- acronym exact match;
- normalized-name Jaro-Winkler similarity;
- core-name Jaro-Winkler similarity;
- normalized Levenshtein ratio;
- token Jaccard similarity;
- name-length ratio.

### 5.3 Address features

- exact normalized-address match;
- address Jaro-Winkler similarity;
- normalized address Levenshtein ratio;
- address-token Jaccard similarity;
- address-length ratio.

### 5.4 Numeric, postal and quality features

- exact address-number signature;
- address-number Jaccard similarity;
- address-number conflict;
- exact house number and house-number conflict;
- exact postal code and postal-code conflict;
- missing-name indicator;
- missing-address indicator.

Explicit conflict features are important for a precision-heavy objective. Two records can have similar names while contradictory house numbers or postal codes provide strong evidence against a match.

Feature-importance analysis showed that address token similarity, candidate pass count, name Jaro-Winkler similarity, address-number overlap, rank distance, and the postal/house retrieval pass carried substantial gain.

## 6. Training strategy

### 6.1 Deterministic split

S1 entities are assigned to model-training or validation partitions through a deterministic hash split. All candidate pairs belonging to an S1 entity remain in the same partition, preventing pair-level leakage between train and validation.

### 6.2 Class imbalance control

Candidate sets contain many more negative than positive pairs. All available positive model-training pairs are retained, while negative pairs are deterministically hash-sampled to a maximum of approximately three million for initial training. Ordinary negative pairs receive weight `2.0`.

### 6.3 Hard-negative mining

A 150-tree baseline LightGBM model is first trained. It then scores the remaining model-training negatives. Negatives receiving probability at least `0.20` are difficult examples that resemble real matches. Up to two million of these hard negatives are retained, assigned weight `3.0`, and added to the final training set.

This directly focuses learning on false-positive patterns, which is valuable because F0.5 is precision-heavy.

### 6.4 Final LightGBM model

The final binary LightGBM classifier uses:

| Parameter | Value |
|---|---:|
| Trees | 450 |
| Learning rate | 0.035 |
| Leaves | 63 |
| Minimum child samples | 250 |
| Maximum bins | 255 |
| Row subsampling | 0.90 |
| Column subsampling | 0.90 |
| L1 regularization | 0.3 |
| L2 regularization | 3.0 |
| Seed | 42 |

The saved text model can be loaded directly by LightGBM without retraining, provided the exact stored feature order is used.

## 7. Threshold selection and post-processing

### 7.1 Pair thresholds

Validation predictions are checkpointed in 32 shards. A coarse and fine global threshold search is followed by a joint S2/S3 search. The best base validation thresholds were approximately:

- S2: `0.37`;
- S3: `0.37`.

The base validation result around this operating point was macro F0.5 `0.89473`, pair precision `0.97614`, and pair recall `0.80689` before the final uniqueness correction was fully evaluated.

### 7.2 Target-side uniqueness

Source 1 is deduplicated, so one S2/S3 target should generally not be assigned to several different S1 entities. The post-processing stage therefore keeps a candidate target only for its strongest S1 claimant. A minimum winning probability margin of `0.02` is used to reject ambiguous ownership.

With all 32 validation-prediction shards present, this raised validation macro F0.5 to approximately **0.91356** while retaining precision near `0.978`.

Further node pruning was tested. Limiting each S1 to eight matches changed validation macro F0.5 only from about `0.913558` to `0.913562`; stronger top-K or probability-gap pruning reduced recall and did not provide a meaningful improvement. It was therefore not adopted as a major final stage.

### 7.3 Country-aware calibration

Training contains US and India, while test additionally contains France. Validation confirmed that `0.37` remained best for both observed training countries:

- India validation macro F0.5 at `0.37`: `0.886800`;
- US validation macro F0.5 at `0.37`: `0.931403`.

France could not be calibrated from labeled training data. Public feedback showed that a much more conservative threshold improved generalization:

| France threshold | Public score |
|---:|---:|
| 0.40 | 0.901 |
| 0.55 | 0.903 |
| 0.70 | 0.904 |
| 0.85 | **0.905** |
| 0.90 | **0.905** |
| 0.98 | 0.900 |

The chosen configuration keeps US/India at `0.37`, France at `0.85`, and the target uniqueness margin at `0.02`. Threshold `0.90` tied publicly but discards more potential true links, so `0.85` is the safer of the tied choices. Increasing only the France margin to `0.10` reduced the public score to `0.903`.

Because these France choices use public-leaderboard feedback, they should be treated as calibration experiments rather than unbiased validation estimates.

## 8. Inference and output generation

Test inference repeats exactly the training transformations:

1. normalize S1/S2/S3;
2. run all eight Sorted Neighbourhood passes per country;
3. union candidates into 32 shards;
4. build the same 33 features in the saved order;
5. load `enhanced_lightgbm.txt` and score every shard;
6. apply thresholds and uniqueness post-processing;
7. stream grouped results into TSV files.

Streaming output generation is necessary because globally aggregating the complete candidate data caused memory exhaustion in Colab. The final writer processes one deterministic S1 bucket at a time and verifies that exactly one output row exists for every test S1 entity.

The two required artifacts are:

- `matching_results.tsv`, containing `source1_entity_id` and `matched_entity_ids`;
- `candidate_pairs.tsv`, containing `source1_entity_id` and `candidate_entity_ids`.

Empty strings are preserved for S1 records with no predicted matches or no candidates.

## 9. Experimental progression and insights

| System version | Public score | Main change |
|---|---:|---|
| Initial Sorted Neighbourhood | 0.844 | Simple candidate passes and pair classifier |
| Enhanced model | 0.875 | Better normalization, more passes/features, hard negatives |
| Uniqueness-aware output | 0.900 | Resolve competing S1 claims for each target |
| Country-calibrated final system | **0.905** | Conservative France threshold |

Important conclusions from the experiments:

1. **Candidate generation and classification are separate.** Sorted Neighbourhood retrieves plausible pairs; LightGBM decides which pairs are matches.
2. **Multiple weak views are better than one sorting key.** Names, reordered tokens, numeric addresses, acronyms, postal/house information, and address tails recover different corruption patterns.
3. **Retrieval provenance is predictive.** Pass count and rank distance became important classifier signals.
4. **Hard negatives matter for F0.5.** Training on confusing false pairs reduces damaging false merges.
5. **Global threshold tuning was nearly exhausted.** US and India validation both favored `0.37`, while node pruning provided negligible benefit.
6. **Domain shift dominated France.** A country-specific conservative threshold produced the final gains.
7. **The candidate ceiling matters.** With candidate recall near 90% and oracle macro F0.5 near 96.1%, reaching 0.95 would require both substantially better retrieval and an almost perfect classifier/post-processor.

## 10. Reproducibility and fault tolerance

The implementation is designed for unstable Colab sessions:

- normalized files, candidate shards, feature shards, validation predictions, selected test shards, models, metrics, and outputs are checkpointed;
- files are copied through a temporary `.partial` name and atomically renamed;
- completed shards are detected and restored rather than recomputed;
- progress bars and per-stage row counts expose long-running work;
- DuckDB memory, thread count, temporary directory, and insertion-order behavior are explicitly configured;
- output row counts and schemas are validated before submission.

Core reproducibility artifacts in `models/` are:

- `enhanced_lightgbm.txt`;
- `enhanced_validation_metrics.json`;
- `enhanced_threshold_results.csv`;
- `enhanced_feature_importance.csv`.

## 11. Limitations and future work

### Current limitations

- Candidate recall leaves roughly 10% of true links unavailable to the classifier.
- France has no labeled training examples, making its calibration vulnerable to public-leaderboard overfitting.
- Transliteration helps but cannot fully model semantic or multilingual address equivalence.
- The large candidate output requires substantial storage and cannot be committed normally to GitHub.
- Public score does not guarantee the same private-leaderboard ranking.

### Most promising extensions

1. Add a genuinely complementary candidate channel, such as frequency-capped rare-key hash blocking or multilingual embedding nearest neighbours.
2. Measure newly recovered validation truth links before paying the cost of full test inference.
3. If a new channel contributes candidates, retrain the classifier with channel indicators and channel-specific hard negatives.
4. Improve country-aware modeling using labeled examples or a principled out-of-domain calibration set rather than leaderboard feedback.
5. Consider character n-gram or learned multilingual similarity features for spelling and transliteration errors not captured by the current sort keys.

Any extension must continue to obey the competition prohibition on external business-entity lookup.

## 12. Final configuration

The strongest observed submission used:

- eight-pass, country-partitioned Sorted Neighbourhood;
- 32 candidate/feature shards;
- 33 pair features;
- weighted random negatives plus mined hard negatives;
- 450-tree LightGBM classifier;
- US threshold `0.37`;
- India threshold `0.37`;
- France threshold `0.85` (with `0.90` tying publicly);
- target-side uniqueness with margin `0.02`;
- streamed exact TSV generation.

This configuration achieved a best observed public leaderboard macro F0.5 score of **0.905**.
