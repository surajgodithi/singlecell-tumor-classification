# Single-Cell Tumor Classification

This project starts by fine-tuning single-cell foundation transformers to separate tumor from normal cells in colorectal cancer (GSE144735), with the long-term goal of extending to additional cancers.

## Project Goal and Strategy

**Current focus: CRC Leave-One-Donor-Out (LODO) pipeline** — rigorous end-to-end analysis
of which genes drive tumor identity in colorectal cancer before expanding to other tissues.

**Pipeline order:**
```
QC → Tokenization → LODO CV → Gene Ranking → In Silico Perturbation → Aggregation
```

1. **LODO Cross-Validation** (`scripts/lodo_cv.py`) — 6-fold donor-held-out training; one fresh Geneformer checkpoint per fold.
2. **Gene Ranking** (`scripts/gene_ranking_analysis.py`) — expression + attention ranking per fold to reveal which genes the model uses.
3. **In Silico Perturbation** (`scripts/in_silico_perturbation.py`) — remove one gene at a time, measure change in P(Tumor); two-phase: known CRC markers + top 200 attention genes.
4. **Aggregation** (`scripts/aggregate_perturbation.py`) — cross-donor consensus ranked target candidate list. *(Step 4 — coming next)*

Multi-tissue transfer (breast, lung) is deferred until the CRC pipeline produces a validated target list. See `docs/Research_Strategy_Transition.md` for the full rationale.

## Quick Start

### 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

For Colab users: Use a High-RAM runtime. The notebooks will handle dependency installation automatically.

### 2. Data Preparation

**Quality Control** (`notebooks/01_quality_control.ipynb`)
- Downloads GSE144735 data from GEO
- Filters cells and genes using standard QC thresholds
- Selects highly variable genes
- Outputs: `gse144735/processed/gse144735_filtered_raw.h5ad`

**Tokenization** (`notebooks/02_tokenisation.ipynb`)
- Converts gene expression into ranked token sequences
- Each cell becomes a sequence of top expressed genes
- Outputs: Token matrix and metadata in `gse144735/processed/tokens/`

**Train/Val/Test Splits** (`notebooks/03_splits.ipynb`)
- Creates patient-wise data splits (no cell leakage between splits)
- 4 patients for training, 1 for validation, 1 for test
- Outputs: Split indices in `gse144735/processed/tokens/`

### 3. Model Training

**Transformer Fine-tuning** (`scripts/finetune_transformer.py`)
- Wraps Hugging Face `Trainer` to fine-tune a pretrained single-cell transformer (e.g., `geneformer/geneformer`) on the ranked tokens.
- Accepts the NPZ tokens, metadata labels, and donor-wise splits generated in the earlier notebooks.
- Example:
  ```bash
  python scripts/finetune_transformer.py \
    --model-name-or-path geneformer/geneformer \
    --model-vocab /path/to/geneformer_gene_vocab.tsv \
    --output-dir outputs/geneformer_finetune
  ```
- The script writes validation/test metrics to `metrics.json` inside the output directory and saves the best checkpoint for downstream analysis.
- `model_vocab` in `configs/finetune.yaml` can point to either a TSV (gene symbol -> token id) or Geneformer's pickled `token_dictionary_gc104M.pkl`; the script auto-detects the format and remaps dataset genes accordingly. Provide `model_gene_name_dict` (e.g., Geneformer's `gene_name_id_dict_gc104M.pkl`) so symbols can be translated to the pretrained model's identifiers when a direct lookup is missing.
- For every new dataset, capture both the rank-based baseline (`scripts/rank_nb_baseline.py`) and the transformer fine-tune metrics (using either the original Geneformer checkpoint or a previously fine-tuned one) so improvements are always benchmarked per dataset. The training script records the best-performing checkpoint path in `outputs/<run>/best_checkpoint.txt`, which the evaluation script uses automatically.
- Recommended per-dataset workflow:
  1. Run `scripts/rank_nb_baseline.py` on that dataset's tokens/splits and log the donor-wise metrics.
  2. Fine-tune Geneformer starting from the original Hugging Face checkpoint and compare against the baseline.
  3. Before updating a multi-cancer checkpoint, evaluate it zero-shot on the new dataset's test split to measure cross-dataset generalization.
  4. Continually fine-tune the latest checkpoint on the new dataset, compare against both the baseline and the fresh fine-tune, and record whether continual learning improved results.
- To skip long CLI commands, edit `configs/finetune.yaml` with your preferred model/checkpoint paths and simply run `python scripts/finetune_transformer.py`; the script auto-loads that config (or pass `--config path/to/file.yaml` for alternates).
- **Checkpoint Evaluation** (`scripts/evaluate_transformer.py`)
  - After training, quickly evaluate val/test splits (and capture per-class precision/recall/F1) either by editing `configs/eval.yaml` and running `python scripts/evaluate_transformer.py`, or by specifying paths explicitly:
    ```bash
    python scripts/evaluate_transformer.py \
      --tokens-dir gse144735/processed/tokens \
      --model-path outputs/geneformer_colon/best-model \
      --model-vocab /content/Geneformer/geneformer/token_dictionary_gc104M.pkl \
      --model-gene-name-dict /content/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl \
      --output-json outputs/geneformer_colon/eval_metrics.json
    ```
  - The script reuses the donor splits, prints accuracy/F1/AUROC for each split you request (default val/test), and writes the metrics JSON so you can compare against baselines at a glance.
- **Binary Tumor vs. Normal variant:** since downstream datasets often lack a Border label, the metadata now includes a `BinaryClass` column where Border cells are merged into Normal. Re-run `scripts/rank_nb_baseline.py` with `--label-column BinaryClass` and set `label_column: BinaryClass` in the fine-tune/eval configs to train the Tumor-vs-Normal checkpoint that future cancers will inherit.
  - The training/eval/baseline scripts will auto-derive `BinaryClass` from the original `Class` column if it is missing (Border -> Normal), so you do not need to retokenize or edit the TSV manually on new machines.

- **Tree-Based Baselines** (`scripts/tree_baseline.py`)
  - Converts ranked tokens into dense inverse-rank features over the top-`k` most frequent genes (default 2,000) and feeds them into classical ensembles.
  - Supports `--model-type random_forest`, `hist_gb` (sklearn HistGradientBoosting), or `xgboost`. Each run writes donor-wise metrics comparable to the Naive Bayes output.
  - Example (binary CRC run):
    ```bash
    python scripts/tree_baseline.py \
      --label-column BinaryClass \
      --model-type hist_gb \
      --output-json baselines/gse144735_tree_histgb_binary_metrics.json
    ```
- **Shallow MLP Baseline** (`scripts/mlp_baseline.py`)
  - Builds the same inverse-rank dense features and trains a lightweight PyTorch MLP (default hidden layers 512x256) with early stopping on the validation donor.
  - Example:
    ```bash
    python scripts/mlp_baseline.py \
      --label-column BinaryClass \
      --output-json baselines/gse144735_mlp_binary_metrics.json
    ```
  - Useful for demonstrating transformer gains over both classical ensembles and small neural nets.

## Results – GSE144735 (Colorectal, Binary Tumor vs. Normal)

Donor-wise splits: train KUL21/KUL28/KUL30/KUL31, val KUL19, test KUL01. All models use `BinaryClass` labels (Border merged into Normal).

### Validation (KUL19 — hardest donor shift)

| Model | Accuracy | Macro F1 | Macro AUC | Tumor Recall |
|---|---|---|---|---|
| Rank Naive Bayes | 0.626 | — | 0.645 | 0.030 |
| XGBoost | 0.582 | 0.526 | 0.643 | 0.316 |
| HistGradientBoosting | 0.582 | 0.566 | 0.639 | 0.514 |
| Shallow MLP | 0.637 | 0.519 | 0.656 | 0.188 |
| **Finetuned Geneformer** | **0.673** | **0.645** | **0.727** | **0.516** |

### Test (KUL01)

| Model | Accuracy | Macro F1 | Macro AUC | Tumor Recall |
|---|---|---|---|---|
| Rank Naive Bayes | 0.708 | — | 0.719 | 0.264 |
| XGBoost | 0.748 | 0.609 | 0.819 | 0.241 |
| HistGradientBoosting | 0.767 | 0.695 | 0.825 | 0.443 |
| Shallow MLP | 0.659 | 0.612 | 0.696 | 0.490 |
| **Finetuned Geneformer** | **0.670** | **0.663** | **0.789** | **0.827** |

Geneformer's key advantage is Tumor Recall — identifying nearly twice as many tumor cells as the best classical model (0.827 vs. 0.443), which matters most for downstream target identification.

## Dataset

**GSE144735**: Single-cell RNA-seq from 6 colorectal cancer patients
- 27,414 cells after QC
- 3 tissue types: Normal, Border, Tumor
- 6 donors (KUL01, KUL19, KUL21, KUL28, KUL30, KUL31)

**GSE131907 (Lung)**: Single-cell RNA-seq from 58 patients (multiple tumor/normal/metastatic sites)
- QC/tokenization/splits complete; tokens/splits under `gse131907/processed/tokens/` (80/10/10 patients: train 46, val 5, test 7).
- Baselines (BinaryClass): NB test acc 0.690/macro AUC 0.753 (Normal F1 low); HistGB test acc 0.889/macro F1 0.843/macro AUC 0.970; XGBoost test acc 0.875/macro F1 0.817/macro AUC 0.964; MLP test acc 0.826/macro F1 0.758/macro AUC 0.906.
- Zero-shot CRC hub (focal2) on lung: val acc 0.488/macro F1 0.433/macro AUC 0.376; test acc 0.696/macro F1 0.659/macro AUC 0.728 (remap matched ~16.9k genes; ~38.5% missing).
- Lung fine-tunes next: base-start run using CRC focal2-style hyperparams adjusted for lung balance (Normal 1.2/Tumor 1.0, focal_gamma 1.5, token_mask_prob 0.01, mixup off, class_donor). CRC-start run will reuse the focal2 checkpoint once a full checkpoint export is available.

## Notes

- The tokenization uses a vocabulary of 24,471 genes
- Each cell is represented by up to 2,048 top-expressed genes
- All splits are patient-wise to ensure generalization
