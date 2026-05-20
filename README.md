# Single-Cell Tumor Classification for Lung Adenocarcinoma Target Identification

This project fine-tunes Geneformer on single-cell RNA-seq from lung adenocarcinoma (GSE131907) to learn a malignant-vs-normal epithelial cell discriminator, then uses **in silico perturbation** to identify which genes the model relies on for that decision. The output is a ranked candidate list of genes whose removal most strongly shifts the model's prediction — a hypothesis-generating tool for downstream experimental target validation.

## Goal

> Identify load-bearing genes for tumor identity in lung adenocarcinoma — which genes, when removed from a cell's expression profile, most reliably shift its predicted identity from Tumor → Normal across patients, and which of those genes have prior causal evidence?

The transformer is a *tool* to prioritize 25,000+ genes down to a tractable testable list. The **ranked candidate gene list** — not the classification metrics — is the deliverable.

## Dataset — GSE131907 (Lung Adenocarcinoma)

Kim et al., *Single-cell RNA sequencing demonstrates the molecular and cellular reprogramming of metastatic lung adenocarcinoma*, *Nature Communications* 2020.

| | Value |
|---|---|
| Total cells in raw dataset | 208,506 |
| Patients after labeling | 26 (17+ contribute Tumor cells, 11 contribute Normal) |
| **Labeled cells used** | **35,757** — 32,054 Tumor + 3,703 Normal |
| Tumor label | `Cell_subtype == "Malignant cells"` **OR** (`Sample_Origin == "tLung"` AND `Cell_type.refined == "Epithelial cells"`) |
| Normal label | `Sample_Origin == "nLung"` AND `Cell_type.refined == "Epithelial cells"` |
| Cells dropped | 172,749 (immune, stromal, metastasis-microenvironment, lymph-node-resident — not the comparison of interest) |
| Split | Patient-level 70/15/15 (18 / 4 / 4 patients) |

The cell-level `Cell_subtype` annotation (authors used clustering + CNV analysis to identify malignant cells directly) eliminates the cell-composition confound that plagues sample-level labels. Both Tumor and Normal pools are restricted to epithelial cells — the lineage of origin for ~85–90% of lung adenocarcinomas — so the classification task is *malignant epithelial vs normal epithelial*, not the easier and less informative *cancer-sample vs normal-sample*.

## Pipeline

```
QC → Tokenization → Patient-level split → Fine-tune (Geneformer-V2-104M)
   → Gene Ranking (raw expression + log2 fold change + attention)
   → In Silico Perturbation (single + knockout sweep + pairwise interaction)
   → External Validation (planned: COSMIC / DepMap / TCGA-LUAD)
```

| Stage | Script |
|---|---|
| QC | `scripts/lung_qc_prep.py` — download, chunked-load (208k-cell-wide TSV), scanpy QC + HVG |
| Tokenization | `scripts/lung_tokenize.py` — labeling rule above, ranked gene tokens (top 2048 / cell) |
| Splits | `scripts/lung_split.py` — patient-level 70/15/15, stratified by `{tumor_only, normal_only, both}` |
| Training | `scripts/lung_train.py` + `configs/lung_train.yaml` |
| Eval (standalone) | `scripts/evaluate_transformer.py` + `configs/eval.yaml` — runs val + test on the best checkpoint |
| Gene ranking | `scripts/gene_ranking_analysis.py` + `configs/lung_perturbation.yaml` — three views: raw expression, log2 fold change (Wilcoxon DEA), and attention |
| Perturbation | `scripts/in_silico_perturbation.py` + `configs/lung_perturbation.yaml` — discovery candidates blended from attention + logFC |
| Aggregation | `scripts/aggregate_perturbation.py` (cross-experiment combining) |

## Perturbation phases

`in_silico_perturbation.py` runs four phases on each held-out cell pool:

1. **Phase 1a — Known drivers** (`phase="known"`): lung adenocarcinoma marker panel (EGFR, KRAS, ALK, TP53, STK11, KEAP1, NF1, RBM10, MET, ERBB2). Sanity check — if the model learned real cancer biology, these should produce strong ΔP(Tumor).
2. **Phase 1b — Negative controls** (`phase="housekeeper"`): housekeeping genes (ACTB, GAPDH, RPL13A, B2M, HPRT1, PPIA, TBP, UBC, YWHAZ, SDHA). Calibrates the noise floor — should produce near-zero |delta|.
3. **Phase 2 — Discovery** (`phase="discovery"`): 200 candidate genes from a **blend** of attention ranking and classical differential expression. 150 slots come from the top-attention genes; 50 slots are reserved for genes high in log2 fold change that aren't already in attention's top 150 (walk-and-skip allocator — overlaps don't cost slots; the script walks deeper into the logFC ranking until 50 truly novel genes are filled). Both perspectives contribute their full share. Where novel candidates surface — including driver-program genes that attention alone might miss because attention can be a noisy explainer (Jain & Wallace 2019).
4. **Phase 3 — Knockout sweep** (`phase="sweep_K{k}"`): cumulative perturbation of top-K hits, K=1..10. Curve shape reveals pathway redundancy — plateau means the model uses ~K dominant signals; linear decrease means independent contributions.
5. **Phase 4 — Pairwise interaction matrix** (`phase="pairwise"`): all `N·(N−1)/2` pairs from top-20 single-gene hits, with epistasis term `ε = Δ_ij − (Δ_i + Δ_j)`. Reveals synergy (`ε > 0`, candidate combo targets) or pathway redundancy (`ε < 0`).

For every gene, two delta values are recorded:
- `delta` — averaged over **all** cells of the class
- `delta_present` — averaged **only** over cells where the gene was actually in the input sequence (biologically meaningful for sparsely-expressed but locally important genes; the primary signal we sort on)

## Setup

```bash
# Python 3.13.5 is pinned (.python-version); install via pyenv or matching
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Geneformer-V2-104M weights — requires git-lfs (brew install git-lfs && git lfs install)
git lfs pull
```

On Apple Silicon the xgboost baseline additionally needs OpenMP: `brew install libomp`.

## Run the full pipeline

```bash
# 1. Download + QC + write filtered h5ad (~30 min, downloads ~411 MB)
python scripts/lung_qc_prep.py

# 2. Apply labeling rule + tokenize (~2 min)
python scripts/lung_tokenize.py

# 3. Patient-level train/val/test split (seconds)
python scripts/lung_split.py

# 4. Train (5–8 epochs, ~7-9 h per epoch on Apple Silicon MPS at L=2048; usually
#    early-stops within 2-4 epochs due to clean labels)
python scripts/lung_train.py

# 5. (Optional) standalone eval on the best checkpoint
python scripts/evaluate_transformer.py

# 6. Gene ranking (expression + logFC + attention) on held-out test patients
python scripts/gene_ranking_analysis.py

# 7. In silico perturbation (Phases 1-4)
python scripts/in_silico_perturbation.py
```

Default configs are `configs/lung_perturbation.yaml` for gene ranking + perturbation and `configs/eval.yaml` for standalone eval — no flags required.

Outputs:
- `outputs/lung/` — Hugging Face training checkpoints
- `results/lung/lung_summary.tsv` — val/test summary (from training)
- `results/lung/lung_val_metrics.json`, `lung_test_metrics.json` — per-class precision/recall/F1 (from training)
- `results/lung/lung_eval_metrics.json` — standalone eval on best checkpoint
- `results/lung/lung_test_attention_genes.tsv` — gene attention ranking
- `results/lung/lung_test_logfc_genes.tsv` — gene log2 fold-change ranking with Wilcoxon p-values
- `results/lung/lung_test_combined_ranking.png` — 3-column expression / logFC / attention dot-plot
- `results/lung/lung_test_perturbation.tsv` — all-phase perturbation results

## Current results

Best checkpoint selected by validation macro F1 (early-stopped at epoch 1; subsequent epochs overfit per the val-loss trajectory). Standalone eval via `scripts/evaluate_transformer.py`:

| Split | n_cells | Accuracy | Macro F1 | Macro AUC | Tumor R | Normal R |
|---|---|---|---|---|---|---|
| Val | 3,988 | 0.980 | 0.971 | 0.997 | 0.977 | **0.989** |
| Test | 10,778 | 0.986 | 0.957 | 0.999 | 0.986 | **0.989** |

Both classes recall ≈ 0.99 — the `class_donor` weighted sampler + `class_weights: Normal=4.0` configuration prevents the model from collapsing to the majority Tumor class despite the ~10:1 dataset imbalance. The clean cell-level labels make the classification task converge quickly (epoch 1 = best by val); the analytical value of the project is in what the perturbation phases reveal about *which* features the model is using to discriminate.

For context, classical baselines on the same splits and labels:

| Baseline | Test Macro F1 | Test Tumor R | Test Normal R |
|---|---|---|---|
| Rank-NB | 0.88 | 0.98 | 0.77 |
| HistGradientBoosting | 0.96 | 0.99 | 0.95 |
| XGBoost | 0.96 | 0.99 | 0.93 |
| Shallow MLP | 0.86 | 0.96 | 0.89 |
| **Geneformer-V2-104M** | 0.96 | 0.99 | **0.99** |

Tree ensembles match Geneformer on macro F1; the transformer's edge is concentrated in Normal recall (the underrepresented class). The reason to use the foundation model isn't the headline accuracy — it's the perturbation framework (knockout sweeps, pairwise epistasis, attention + logFC blend) layered on top, which trees can't natively support.

## Implementation notes

- **Compute target:** Apple Silicon (M5, 32 GB unified memory). Speed/memory stack — length-bucketed sampler, dynamic padding, gradient checkpointing, SDPA — makes Geneformer-V2-104M training at L=2048 feasible on consumer hardware. Per-step pace varies with PyTorch version (~33 s/step on 2.11, ~46 s/step on 2.12 — MPS attention kernels differ between releases). Wrap long-running scripts in `caffeinate -i` to prevent laptop sleep.
- **Patient ID extraction:** `re.search(r"(\d+)", sample_name)` — so `LUNG_T28` + `LUNG_N28` + `EBUS_28` all collapse to patient `28` and stay in the same split.
- **Tokenization:** per cell, top 2048 expressed genes ranked descending by raw counts, mapped to Geneformer's 104M-token vocabulary via gene symbol → ID lookup with an alias table for missing matches (~62% direct vocabulary coverage on lung; the ~38% missing are mostly pseudogenes, rarely-expressed lncRNAs, and recently-annotated transcripts not in Geneformer's pretraining corpus — they contribute very little to the expression signal of any individual cell).
- **All splits are patient-level** — no cell-level leakage between train/val/test. The 70/15/15 fraction is by patient count (18/4/4); cell counts (~21k/4k/11k) differ because some patients yield more cells than others, particularly in the test split.
- **Discovery candidate blend** (Phase 2): walk-and-skip allocator reserves 150 slots for attention top-N and 50 slots for novel logFC genes (walks the logFC ranking, skips any gene already in attention's top 150, fills the reserved slots with the next non-overlapping candidates). Both perspectives contribute their full share even when rankings agree heavily.
