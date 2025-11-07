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


