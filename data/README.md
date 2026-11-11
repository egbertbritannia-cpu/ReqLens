# Data

The active dataset for the current paper scope is:

```text
data/exp/promise_exp.csv
```

## Active Dataset

### PROMISE Expanded

- File: `exp/promise_exp.csv`
- Rows: 969
- Columns: `ProjectID`, `RequirementText`, `class`
- Labels in file: `F`, `SE`, `US`, `PE`, `LF`, `O`, `A`, `MN`, `SC`, `FT`, `L`, `PO`

Legacy notebooks use this file directly and create a stratified 70/15/15 train/validation/test split at runtime. The ICIIT 2027 K1 evaluation uses both random stratified and project-disjoint splits, without augmentation.

The public GitHub file and the frozen K1 artifact copy contain the same ordered CSV records. The K1 checker verifies the exact file bytes, so a direct GitHub download does not match its expected hash.

| Copy | Line endings | SHA-256 |
|---|---|---|
| [Public CSV at commit 7889d82](https://raw.githubusercontent.com/egbertbritannia-cpu/ReqLens/7889d8205370da91d759441cf75a62219298b7f3/data/exp/promise_exp.csv) | LF | `b8762702ebb76cb34d7005a3952e918ff2f96bc1f98c580294806fba775036a4` |
| Frozen K1 copy in the unreleased reproduction package | CRLF | `f1786d5fc1da6714d0c4886727d98668d707f865dc8dde3988eb735711dab0b5` |

## Directory Meaning

```text
data/
├── exp/                 # Active experiment dataset
├── raw/                 # Source/archive files
├── processed/           # Legacy merged-corpus split, not active
├── processed_ablation/  # Legacy ablation split, not active
└── paraphrased/         # Optional future robustness data
```

## Current Protocol

The current repository version does not use the previous merged FR-heavy corpus. It uses `promise_exp.csv` as the single source of data for the paper experiments.

The `class` column is preserved as provided by PROMISE expanded. No external FR-only dataset is merged into this protocol.
