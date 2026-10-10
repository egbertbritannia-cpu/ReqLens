# SERF implementation audit: routing and prediction provenance

**Scope:** Code inspection of the two `without_augmentation` notebooks on the repository `main` branch, plus inventory of tracked files. This is a technical audit, **not** a project-status tracker or a completed experimental comparison.

**Evidence / source of truth:**
- `code/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb` (Git blob `5634e03cfd2fec9630283833dd60a125f3bd3fe6`).
- `model-training/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb` (Git blob `695b8a723c2bd669c1263c494fea7f6e18b127a5`).
- `data/README.md` and `data/exp/promise_exp.csv`.
- `tests/test_serf_router_contract.py`, `experiments/routing/route_ablation.py`, `tests/test_route_ablation.py`, `tests/test_serf_margin_redundancy.py`.
- `experiments/routing/prediction_cache_contract.py` and `tests/test_prediction_cache_contract.py`.

## 1. What the two notebooks actually do

| Property | `code/` development notebook | `model-training/` notebook |
| --- | --- | --- |
| Selector | ELECTRA logits, optionally temperature-calibrated via validation | Separately trained ELECTRA-only classifier, uncalibrated softmax |
| Hard-path prediction | Calibrated ELECTRA / RoBERTa **probability interpolation** with alpha | **Separately trained ELECTRA–RoBERTa fusion classifier** |
| Routing confidence | Maximum of calibrated selector probabilities | Maximum of uncalibrated selector probabilities |
| Margin | Largest minus second-largest probability | Largest minus second-largest probability |
| Risky-class check | Selector argmax belongs to `R` | Selector argmax belongs to `R` |
| Fusion execution | Development code can calculate both experts for diagnostics/tuning | Calls fusion model only for hard subset during selective evaluation |
| Test-set outcome | Reported summary metrics / aggregated plots | Reported summary metrics / aggregated plots |

The common Boolean logic is `(confidence < tau) OR (margin < delta) OR (predicted_class in R)`.
In the training notebook it is written as the negation of an easy-path conjunction; the two expressions are equivalent. Both use `R={A,L,FT,MN,PO,SC,PE,O,US}`, `tau ∈ {0.70,0.80,0.85,0.90}`, `delta ∈ {0.15,0.20}`. The development implementation also searches alpha. The notebooks set fixed `SPLIT_SEED=42` and `TRAIN_SEEDS=[42,123,2024]`.

**Important:** Do not mix predictions/checkpoints or report routing ablations as if these architectures implement the same fusion expert. Temperature affects confidence/margin values; architecture and calibration must be included in the provenance.

## 2. Proven margin redundancy for the current parameter grid

For a valid probability distribution, if `p1` and `p2` are the largest and second-largest probabilities, then `p2 ≤ 1 − p1`, hence `margin=p1−p2 ≥ 2p1−1`. Whenever `p1 ≥ tau ≥ 0.70`, `margin ≥ 0.40 > delta` for either legacy delta. Therefore `margin < delta` implies `confidence < tau`: **the margin trigger cannot route an additional sample for the historical grid**. This is a mathematical property, not a new Macro-F1 result.

The risky-class trigger can still add samples independently. Whether confidence adds material predictive value beyond class-based routing requires matched ablation on genuine, aligned predictions.

## 3. Current prediction / checkpoint availability

Repository-wide tracked-file inventory on `main` contains the active dataset `data/exp/promise_exp.csv`, raw datasets and notebooks, but **no committed per-sample selector probability tables, matched fusion predictions, trained model checkpoints or serialized validation/test prediction caches**. The training notebook has some embedded notebook outputs, which are **not** a sufficient replacement for aligned per-instance predictions/checkpoints.

The two inspected notebooks read `promise_exp.csv`, then apply `df = df[["RequirementText", "class"]].dropna().reset_index(drop=True)`. This **discards `ProjectID` and the raw CSV row index**; later `train_df`, `val_df`, `test_df` are also reset. In both notebook Dataset classes, `__getitem__` returns tokenized tensors and a label, not a stable source sample identifier. Evaluation DataLoaders are created without shuffle, but positional alignment alone does not prove the same dataset/split/checkpoint was used for both experts.

Consequences:
1. Do not fabricate `sample_id` from post-filtered/test-only array position or assume `ProjectID` alone is globally unique.
2. A future export must preserve `source_row_index` from the original CSV **before** normalization, filtering and splitting; associate it with all validation/test outputs.
3. Export selector probabilities and the fusion prediction for **every** sample under identical label mapping/split/checkpoints, not only those routed by Full SERF.
4. Freeze all thresholds/calibration/alpha using validation data. Never tune ablation parameters using the test set.
5. Save and independently verify dataset SHA-256, checkpoint SHA-256s, notebook commit, architecture, train/split seed, split, label order and temperature/interpolation parameters.

## 4. Safe interface for the eventual prediction export

`experiments/routing/prediction_cache_contract.py` implements `validate_cache(payload)`. A cache payload has:

- `metadata`: `schema_version=1`, `variant` (`development_late_fusion` or `training_separate_fusion`), `dataset_sha256`, `notebook_path`, `notebook_commit_sha`, `split` (`validation`/`test`), `split_seed`, `train_seed`, `label_order`, selector/fusion checkpoint SHA-256s, `fusion_method`, selector/RoBERTa temperature and alpha as appropriate, `threshold_selection_split=validation`, `tau`, `delta`, `risky_labels`.
- `records`: one per original CSV row, containing `source_row_index`, `true_label`, class-aligned `selector_probs` and `fusion_pred`.

The validator rejects missing provenance, incompatible fusion recipes, absent split information, duplicate source rows and malformed probability vectors, then generates a dataset-hash-qualified sample ID. Its returned rows can be passed to `route_ablation.evaluate_cached`.

A schema-valid payload is **not independently validated research evidence**: hashes must correspond to actual artifacts, labels to actual train-time mapping, and sample membership to the original split. The present two notebooks do not yet emit this schema. `IMPLEMENTED_NOT_RUN` on research predictions.

## 5. Verification and next safe action

- The pre-existing V2 routing suite was rerun on locally available repository-matching source: **14/14 CPU tests PASS** (8 ablation + 6 mathematical checks), `py_compile` PASS. Synthetic probability inputs only.
- The new prediction-cache contract has **8/8 CPU tests PASS**, `py_compile` PASS; both Python files were verified against GitHub contents by blob hash. Synthetic manifests/checksum-shaped placeholders only.
- Full pretrained notebook execution, new model training, original trained checkpoint recovery, alignment of real prediction artifacts, and P06–P09 benchmark scores: **NOT RUN / NOT AVAILABLE**.

**Next safe engineering step:** Implement an opt-in, non-destructive export hook after retrieving/verifying the original checkpoints (or after explicit authorization for a new run). Preserve `source_row_index` before filtering; export full validation/test selector and fusion outputs; run `validate_cache`, then evaluate fixed ablation modes with repeated matched-random draws and per-seed paired analysis. If source checkpoints remain absent, report **BLOCKED** instead of treating historical aggregate notebook outputs as predictions.

Do not mark routing experiments complete merely because their validator/tests exist.

## 6. Dataset split reconstruction check (2026-10-10)

**Method and limitation:** The private original `data/exp/promise_exp.csv` was fetched and parsed using the connected GitHub API (Git blob `9f91b2c5a761e648311bddf306f9cd67ba7650ff`, 111,660 bytes). The 969 source rows all contain nonempty requirement text and valid class labels, and no class is a singleton. Because the Python runtime could not directly download the private CSV bytes, the **ordered original class column** was projected into a local CSV with nonempty placeholder text and the same raw row positions. The committed `experiments/routing/split_provenance.py` was then run on that projection, preserving every class and source row's stratification position. This is an **exact split-membership reconstruction conditional on the verified nonempty-row property, NOT a direct execution on the original CSV bytes**. The local projection SHA-256 must not be confused with the SHA-256 of the original dataset.

Code/provenance checked: `split_provenance.py`, `tests/test_split_provenance.py`, and the saved output in `model-training/ELECTRA_RoBERTa_SERF_without_augmentation.ipynb` (Git blob `695b8a723c2bd669c1263c494fea7f6e18b127a5`). The script's two sequential stratified `train_test_split` operations use `test_size=0.30`, then `test_size=0.50`, each with `random_state=42`. **All four complete, train, validation and test per-class distributions match the legacy notebook's saved output.**

| Class | Complete | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| A | 31 | 22 | 4 | 5 |
| F | 444 | 311 | 66 | 67 |
| FT | 18 | 13 | 3 | 2 |
| L | 15 | 11 | 2 | 2 |
| LF | 49 | 34 | 7 | 8 |
| MN | 24 | 17 | 4 | 3 |
| O | 77 | 54 | 11 | 12 |
| PE | 67 | 47 | 10 | 10 |
| PO | 12 | 8 | 2 | 2 |
| SC | 22 | 15 | 4 | 3 |
| SE | 125 | 87 | 19 | 19 |
| US | 85 | 59 | 13 | 13 |
| **Total** | **969** | **678** | **145** | **146** |

**Computed sample index order fingerprints:** SHA-256 of ASCII comma-separated, zero-based source row indices for train = `f5f71b6e916f2a713e02931d83fa40a7e6da59ccd43b77032ec564bb29580d00`, validation = `7b65d735dda69c70c1a01b272386839d342c71deb1938ff2969751faf2487b20`, test = `3962006bad5e1fd1cab4fe772b9730be879841bc872fe862695c2644240de4d5`. These are **split-membership fingerprints**, not dataset or checkpoint hashes.

**Checks:** local CPU `unittest` 5/5 PASS for split-provenance suite, `py_compile` PASS. Notebook saved class-distribution cross-check PASS. No original model checkpoints, saved per-sample predictions, GPU inference, P06-P09 ablation or independent full original-file execution took place. Research evidence gate remains **BLOCKED** on original model checkpoints and aligned prediction export. Original scores and manuscript untouched.
