# Beyond Accuracy: Enhancing the Robustness of Software Requirements Classification against Paraphrasing using Paradigm-Complementary Hybrid Transformers

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)


## Research Tracking

**[Mở Research Tracking — Google Sheets](https://docs.google.com/spreadsheets/d/1le9GUaezNO4KWGSYG__CJDPCaszxJ7c8PzinLgGXFLc/edit)**

Kế hoạch cải thiện paper SERF sau phản biện EAI FISAT 2026, do **Quách Thế Dương** thực hiện. Quản lý công việc, tiến độ, phản biện, thí nghiệm và các mốc nghiên cứu trực tiếp trên Google Sheets — không cập nhật trạng thái trùng lặp bằng Markdown.

## Overview

Current AI models achieve high accuracy on static requirement datasets, yet they remain brittle when stakeholders rephrase the same intent using different sentence structures — a common reality in Agile environments. This project proposes and compares multiple **Hybrid Transformer architectures** for software requirements classification (FR vs. NFR sub-types), with a focus on both accuracy and robustness.

The core contribution is the **SERF (Selective Ensemble Routing Framework)** — a hybrid ELECTRA+RoBERTa architecture that dynamically routes each input through either a fast single-encoder path or a full fusion path, depending on prediction confidence.

## Research Questions

| # | Research Question | Focus |
|---|-------------------|-------|
| **RQ1** | Which model achieves the highest performance for requirements classification? | Compare hybrid models against single-encoder baselines |
| **RQ2** | How does model robustness change when software requirements are paraphrased? | Measure consistency via a Robustness Score on paraphrase variants |
| **Discussion** | What linguistic features cause models to misclassify? | Exploratory XAI analysis on mispredictions |

## Architecture: SERF (Selective Ensemble Routing Framework)

The proposed **ELECTRA+RoBERTa SERF** routes each input based on ELECTRA's prediction confidence:

```
Input Text
    |
    v
ELECTRA Confidence Router
(tau threshold + delta margin + risky-class check)
    |
    +-- [High Confidence & Non-Risky] --> ELECTRA-only path
    |       ELECTRA Encoder -> LayerNorm -> Dense(768) -> GELU -> Dropout -> Classifier
    |
    +-- [Low Confidence or Risky Class] --> Fusion path
            ELECTRA Encoder --+
                              +--> Concat(1536) -> LayerNorm -> FC(768) -> GELU -> Dropout -> Classifier
            RoBERTa Encoder --+
    |
    v
Softmax -> Final Prediction (multi-class NFR type / FR)
```

| SERF Parameter | Search Space |
|----------------|-------------|
| tau (confidence threshold) | {0.70, 0.80, 0.85, 0.90} |
| delta (margin threshold) | {0.15, 0.20} |
| ELECTRA model | `google/electra-base-discriminator` |
| RoBERTa model | `roberta-base` |
| Frozen layers | Lower 8 transformer layers |
| Learning rate | 2e-5 |
| Epochs | 30 |

> See architecture diagrams in [`outputs/architecture/`](outputs/architecture/) and [`notebook_from_paper/SERF_architecture_numeric.drawio`](notebook_from_paper/SERF_architecture_numeric.drawio).

## Models Implemented

| Model | Type | Notes |
|-------|------|-------|
| BERT-base | Single Encoder | Transfer learning baseline |
| RoBERTa-base | Single Encoder | Strong TL baseline |
| ModernBERT | Single Encoder | Ablation |
| ELECTRA-base | Single Encoder | Ablation |
| BERT + RoBERTa Hybrid | Naive Fusion | Concatenation-based hybrid |
| ModernBERT + ELECTRA Hybrid | Gated Fusion | Adaptive gated fusion |
| **ELECTRA + RoBERTa SERF** | **Selective Ensemble** | **Proposed model** |
| ELECTRA + RoBERTa SERF (w/ augmentation) | Selective Ensemble | Data augmentation variant |

## Dataset

The active dataset for all current experiments is **PROMISE Expanded**:

| Property | Value |
|----------|-------|
| File | `data/exp/promise_exp.csv` |
| Records | 969 |
| Columns | `ProjectID`, `RequirementText`, `class` |
| Labels | `F` (Functional), `SE`, `US`, `PE`, `LF`, `O`, `A`, `MN`, `SC`, `FT`, `L`, `PO` |
| Split | Stratified 70 / 15 / 15 (train / validation / test) at runtime |

Raw source files (for reference) are stored in `data/raw/`:
- `PROMISE-relabeled-NICE.csv`
- `promise_exp.csv`
- `dcai24_src_dataset.xlsx`
- `EARS Functional Requirements Complete Dataset.xlsx`

> **Note:** The current paper scope uses **only** `data/exp/promise_exp.csv`. The broader multi-dataset merged corpus (DCAI24, EARS) from the original plan is preserved in `data/raw/` but is not active for current experiments.

## Project Structure

```
rbl-requirements-classification/
|-- code/                             # Exploratory / development notebooks
|   |-- Bert.ipynb
|   |-- ELECTRA.ipynb
|   |-- ModernBERT.ipynb
|   |-- RoBerta_origin_tested.ipynb
|   |-- Hybrid_modernBERT_ELECTRA.ipynb
|   |-- Hybrid_BERT_RoBERTa.ipynb
|   |-- ELECTRA_+_RoBERTa_hybrid.ipynb
|   |-- ELECTRA_RoBERTa_SERF_with_augmentation.ipynb
|   `-- ELECTRA_RoBERTa_SERF_without_augmentation.ipynb
|
|-- model-training/                   # Full training runs (on GPU / Colab)
|   |-- Bert.ipynb
|   |-- ELECTRA.ipynb
|   |-- ModernBERT_noFreeze.ipynb
|   |-- RoBerta_more_specific_data.ipynb
|   |-- Hybrid_modernBERT_ELECTRA.ipynb
|   |-- Hybrid_modernBERT_ELECTRA_2e_1config.ipynb
|   |-- Hybrid_BERT_RoBERTa.ipynb
|   |-- ELECTRA_+_RoBERTa_hybrid_new.ipynb
|   |-- ELECTRA_RoBERTa_SERF_with_augmentation.ipynb
|   `-- ELECTRA_RoBERTa_SERF_without_augmentation.ipynb
|
|-- notebook_from_paper/              # Reference notebooks & architecture diagrams
|   |-- Bert.ipynb
|   |-- RoBerta.ipynb
|   |-- Hybrid_BERT_RoBERTa.ipynb
|   |-- Hybrid_BERT_RoBERTa_without_augmentation.ipynb
|   |-- SERF_architecture_numeric.drawio
|   `-- SERF_workflow_from_image.drawio / .png
|
|-- data/
|   |-- exp/                          # Active experiment dataset
|   |   `-- promise_exp.csv
|   `-- raw/                          # Source/archive files
|       |-- PROMISE-relabeled-NICE.csv
|       |-- promise_exp.csv
|       |-- dcai24_src_dataset.xlsx
|       `-- EARS Functional Requirements Complete Dataset.xlsx
|
|-- outputs/
|   |-- architecture/                 # Architecture diagrams
|   |-- new_modified_analysis/        # SERF architecture docs & analysis scripts
|   `-- docx_to_latex_preview*/       # Paper conversion artifacts
|
|-- docs/                             # Paper drafts & diagrams
|   |-- FullPaper_RBL AI classification.docx
|   |-- RBL_Latex_Paper/              # LaTeX source
|   `-- diagram.drawio
|
|-- requirements.txt
`-- .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU recommended (or Google Colab with T4/A100)

### Local Setup

```bash
git clone https://github.com/egbertbritannia-cpu/rbl-requirements-classification.git
cd rbl-requirements-classification
pip install -r requirements.txt
```

### Running Experiments

1. **Exploratory notebooks** — start from `code/` for development and quick iteration
2. **Full training runs** — use the notebooks in `model-training/` on a GPU runtime
3. **Reference implementations** — see `notebook_from_paper/` for the paper-aligned versions

### Google Colab (Recommended for Training)

1. Upload any notebook from `model-training/` to Google Colab
2. Select **Runtime → Change runtime type → GPU (T4 or A100)**
3. Mount Google Drive if needed and run all cells

## Dependencies

| Category | Packages |
|----------|---------|
| Data & ML | `pandas`, `numpy`, `scikit-learn`, `openpyxl` |
| Deep Learning | `torch>=2.0`, `transformers==4.40.1`, `datasets`, `accelerate` |
| NLP | `nltk`, `spacy` |
| Explainability | `shap`, `captum` |
| Experiment Tracking | `wandb` |
| LLM Baseline | `openai>=1.40` |
| Visualization | `matplotlib`, `seaborn` |

Install all with:
```bash
pip install -r requirements.txt
```

## Key References

- PROMISE NFR dataset: Kurtanovic & Maalej (2017)
- ELECTRA: Clark et al. (2020) — `google/electra-base-discriminator`
- RoBERTa: Liu et al. (2019) — `roberta-base`
- ModernBERT: Warner et al. (2024)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
