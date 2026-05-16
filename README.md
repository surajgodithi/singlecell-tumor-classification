# Single-Cell Tumor Classification

This project fine-tunes single-cell foundation transformers (Geneformer) to separate cancer cells from normal cells using single-cell RNA-seq data, and uses the trained models to identify candidate tumor driver/suppressor genes via in silico perturbation across patients.

## Active dataset (2026-05-15 onward)

**GSE131907 — Lung Adenocarcinoma** (Kim et al., *Nature Communications* 2020).

| | Value |
|---|---|
| Total cells | 208,506 |
| Patients | ~28 with confident cancer cells, 11 with matched normal lung |
| Cancer-cell label | **Cell-level** (`Cell_subtype == "Malignant cells"`) + tLung epithelial (`tS1`/`tS2`/`tS3`) |
| Normal label | `Sample_Origin == "nLung"` AND `Cell_type.refined == "Epithelial cells"` (3,703 cells across 11 patients) |
| Split strategy | Patient-level 70/15/15 train/val/test |

Why this dataset over GSE144735 (CRC): the lung data has **cell-level malignancy annotations** by the original authors, eliminating the cell-composition confound that limits sample-level labels.

## Project goal

> **Identify load-bearing genes for tumor identity in lung adenocarcinoma** — which genes, when removed from the input, most reliably shift a cell's predicted identity from Tumor → Normal across patients? Validate consistency across held-out patients and cross-reference against external resources (COSMIC, DepMap).

The transformer is a *tool* to prioritize 16,000+ genes down to a tractable testable list. The ranked candidate gene list — not the classification metrics — is the deliverable.

## Pipeline (current)

```
QC → Tokenization → Patient-level train/val/test split → Fine-tune → Gene Ranking
   → In Silico Perturbation (single-gene + knockout sweep + pairwise) → Aggregation
```

| Stage | Script / notebook | Status |
|---|---|---|
| QC | `scripts/lung_qc_prep.py` (mirrors `notebooks/lung_01_quality_control.ipynb`, Colab cells stripped, save step added) | New |
| Tokenization | `notebooks/lung_02_tokenisation.ipynb` — label rule needs update | Needs rewrite |
| Splits | `scripts/lung_split.py` (patient-level 70/15/15) | To write |
| Fine-tune | Adapted from `scripts/lodo_cv.py` to single split | To adapt |
| Gene ranking | `scripts/gene_ranking_analysis.py` | Adapts |
| Perturbation | `scripts/in_silico_perturbation.py` | Adapts + biology-first revisions |
| Aggregation | `scripts/aggregate_perturbation.py` | Adapts |

## Biology-first revisions in progress

Beyond the dataset pivot, several pipeline improvements are being implemented to make the analysis biologically credible rather than just metric-maximizing:

1. **Dilution fix** — perturbation `delta` computed over cells *where the gene is present*, not all cells.
2. **Negative-control genes** — housekeepers in Phase 1 to calibrate the noise floor.
3. **Knockout sweep** — cumulative perturbation of top-1..top-10 hits, not arbitrary K.
4. **Cell-type stratification** — built in to lung via `Cell_type.refined`. Epithelial-only perturbation will be a primary output.
5. **Bootstrap CI** on per-fold deltas for statistical interpretation.
6. **External validation** — `scripts/cross_reference_targets.py` joining final ranked list against COSMIC Cancer Gene Census and DepMap CRISPR essentiality.
7. **Pairwise interaction matrix** (top 20 genes, 190 pairs) — reveals pathway redundancy and synergy.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Lung dataset is downloaded automatically by scripts/lung_qc_prep.py via pooch
# (~410 MB compressed counts + small annotation)
```

The model checkpoint (Geneformer-V2-104M) lives under `Geneformer/` after `git lfs pull`.

## Status (2026-05-15)

- CRC LODO pipeline development is **paused**. KUL01 fold (Tumor recall 0.682, macro_f1 0.7225 at L=2048) is on disk as a pilot demonstrating the pipeline runs end-to-end.
- Lung raw data downloaded (`gse131907/raw/`).
- Lung labeling decision finalized.
- Lung tokenization + split scripts: to be written.
- Lung training: not started.

## Historical results — GSE144735 (CRC)

These were the pilot runs that validated the pipeline before the lung pivot. Kept for reference and to demonstrate the comparison baseline classical methods provide. **Not the active analysis.**

### Validation (KUL19 — pre-LODO hardest donor)

| Model | Accuracy | Macro F1 | Macro AUC | Tumor Recall |
|---|---|---|---|---|
| Rank Naive Bayes | 0.626 | — | 0.645 | 0.030 |
| XGBoost | 0.582 | 0.526 | 0.643 | 0.316 |
| HistGradientBoosting | 0.582 | 0.566 | 0.639 | 0.514 |
| Shallow MLP | 0.637 | 0.519 | 0.656 | 0.188 |
| **Finetuned Geneformer** | **0.673** | **0.645** | **0.727** | **0.516** |

### Test (KUL01 — pre-LODO test donor)

| Model | Accuracy | Macro F1 | Macro AUC | Tumor Recall |
|---|---|---|---|---|
| Rank Naive Bayes | 0.708 | — | 0.719 | 0.264 |
| XGBoost | 0.748 | 0.609 | 0.819 | 0.241 |
| HistGradientBoosting | 0.767 | 0.695 | 0.825 | 0.443 |
| Shallow MLP | 0.659 | 0.612 | 0.696 | 0.490 |
| **Finetuned Geneformer** | **0.670** | **0.663** | **0.789** | **0.827** |

### LODO (KUL01 fold only, 2026-05-15)

| Metric | Value |
|---|---|
| Accuracy | 0.7506 |
| Macro F1 | 0.7225 |
| Macro AUC | 0.8278 |
| Tumor P / R / F1 | 0.593 / 0.682 / 0.634 |
| Normal P / R / F1 | 0.841 / 0.782 / 0.811 |

KUL01 LODO Tumor recall (0.682) fell short of the 0.70 target. Per-fold reliability varies sharply with donor cell counts (KUL31 has only 75 Tumor cells) — one of several factors motivating the lung pivot.

## Notes

- Tokenization uses a per-cell ranked vocabulary of expressed genes (top 2,048 per cell, mapped to Geneformer's 104M-token vocabulary).
- All splits are patient-level to ensure no cell leakage between train and held-out.
