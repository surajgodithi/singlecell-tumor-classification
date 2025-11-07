# Single-Cell Tumor Classification

This project starts by fine-tuning single-cell foundation transformers to separate tumor from normal cells in colorectal cancer (GSE144735), with the long-term goal of extending to additional cancers.

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
- For every new dataset, capture both the rank-based baseline (`scripts/rank_nb_baseline.py`) and the transformer fine-tune metrics (using either the original Geneformer checkpoint or a previously fine-tuned one) so improvements are always benchmarked per dataset.
- Recommended per-dataset workflow:
  1. Run `scripts/rank_nb_baseline.py` on that dataset's tokens/splits and log the donor-wise metrics.
  2. Fine-tune Geneformer starting from the original Hugging Face checkpoint and compare against the baseline.
  3. Before updating a multi-cancer checkpoint, evaluate it zero-shot on the new dataset's test split to measure cross-dataset generalization.
  4. Continually fine-tune the latest checkpoint on the new dataset, compare against both the baseline and the fresh fine-tune, and record whether continual learning improved results.
- To skip long CLI commands, edit `configs/finetune.yaml` with your preferred model/checkpoint paths and simply run `python scripts/finetune_transformer.py`; the script auto-loads that config (or pass `--config path/to/file.yaml` for alternates).
- **Checkpoint Evaluation** (`scripts/evaluate_transformer.py`)
  - After training, quickly evaluate val/test splits (and capture per-class precision/recall/F1) by pointing the script at your tokens directory and checkpoint:
    ```bash
    python scripts/evaluate_transformer.py \
      --tokens-dir gse144735/processed/tokens \
      --model-path outputs/geneformer_colon/best-model \
      --model-vocab /content/Geneformer/geneformer/token_dictionary_gc104M.pkl \
      --model-gene-name-dict /content/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl \
      --output-json outputs/geneformer_colon/eval_metrics.json
    ```
  - The script reuses the donor splits, prints accuracy/F1/AUROC for each split you request (default val/test), and writes the metrics JSON so you can compare against baselines at a glance.

## Dataset

**GSE144735**: Single-cell RNA-seq from 6 colorectal cancer patients
- 27,414 cells after QC
- 3 tissue types: Normal, Border, Tumor
- 6 donors (KUL01, KUL19, KUL21, KUL28, KUL30, KUL31)

## Project Structure

```
gse144735/
â”œâ”€â”€ raw/              # Downloaded GEO files
â””â”€â”€ processed/
    â”œâ”€â”€ *.h5ad        # Filtered AnnData objects
    â””â”€â”€ tokens/       # Tokenized data and splits
notebooks/            # Analysis workflows
```

## Notes

- The tokenization uses a vocabulary of 24,471 genes
- Each cell is represented by up to 2,048 top-expressed genes
- All splits are patient-wise to ensure generalization


