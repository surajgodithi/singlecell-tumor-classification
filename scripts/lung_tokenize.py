#!/usr/bin/env python
"""
Lung (GSE131907) labeling + Geneformer-style tokenization.

Replaces `notebooks/lung_02_tokenisation.ipynb`. Two important departures from the
notebook:

1. **Cell-level labeling** instead of `Sample_Origin`-based.
   The notebook's `class_to_binary` dict (tLung→Tumor, nLung→Normal, etc.) reads
   sample-level labels — meaning *every* cell in a tLung sample (including T cells,
   macrophages, fibroblasts) gets labeled "Tumor." That reintroduces the
   cell-composition confound we pivoted away from CRC to avoid.

   New rule (defended in `Lung_Dataset.md`):
       is_tumor  = (Cell_subtype == "Malignant cells")
                   OR (Sample_Origin == "tLung" AND Cell_type.refined == "Epithelial cells")
       is_normal = (Sample_Origin == "nLung" AND Cell_type.refined == "Epithelial cells")
       drop everything else (immune, stromal, mLN/mBrain/PE microenvironment).

   The `tS1`/`tS2`/`tS3` cells in tLung samples are likely transformed
   AT1/AT2/Club/Ciliated cells — the cancer cells from primary surgical
   resections, just labeled by the authors with a different naming scheme.

2. **No `BinaryClass` collapse of unrelated categories** into a fake "Tumor"
   bucket. We *select* into Tumor/Normal via the rule above, and drop cells
   that don't fit cleanly. This trades total cell count for biological precision.

Output (`gse131907/processed/tokens/`):
    gse131907_gene_rank_tokens.npz   tokens (cells × MAX_GENES), lengths
    gse131907_tokens_metadata.tsv    one row per kept cell:
        Patient | Sample | Sample_Origin | Cell_type | Cell_type.refined |
        Cell_subtype | BinaryClass | token_length
    gene_vocab.tsv                   gene_symbol → dataset token id

Run:
    python scripts/lung_tokenize.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from tqdm import trange

DATA_DIR = Path("gse131907")
PROCESSED_DIR = DATA_DIR / "processed"
TOKEN_DIR = PROCESSED_DIR / "tokens"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

INPUT_H5AD = PROCESSED_DIR / "gse131907_filtered_raw.h5ad"
TOKENS_NPZ = TOKEN_DIR / "gse131907_gene_rank_tokens.npz"
METADATA_TSV = TOKEN_DIR / "gse131907_tokens_metadata.tsv"
VOCAB_TSV = TOKEN_DIR / "gene_vocab.tsv"

MAX_GENES = 2048  # ranked-token sequence length per cell


def extract_patient_id(sample: str) -> str:
    """Pull the numeric patient ID from a sample name; fall back to full string.

    Examples (verified against GSE131907 sample naming):
        LUNG_T28  -> "28"     (surgical primary tumor sample for patient 28)
        LUNG_N28  -> "28"     (matched adjacent normal for patient 28)
        EBUS_28   -> "28"     (bronchoscopy for patient 28; same patient)
        NS_07     -> "07"     (brain met for patient 07)
        BRONCHO_58 -> "58"
    Patient 06 spans LUNG_T06, LUNG_N06, EBUS_06, EFFUSION_06 — all same person.
    """
    m = re.search(r"(\d+)", sample)
    return m.group(1) if m else str(sample)


def apply_labels(adata: sc.AnnData) -> sc.AnnData:
    """Apply the Cell_subtype-based label rule and subset to labelled cells."""
    obs = adata.obs
    required = ["Cell_subtype", "Sample_Origin", "Cell_type.refined", "Sample"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"AnnData.obs missing columns: {missing}")

    is_tumor = (
        (obs["Cell_subtype"] == "Malignant cells")
        | (
            (obs["Sample_Origin"] == "tLung")
            & (obs["Cell_type.refined"] == "Epithelial cells")
        )
    )
    is_normal = (
        (obs["Sample_Origin"] == "nLung")
        & (obs["Cell_type.refined"] == "Epithelial cells")
    )

    binary = pd.Series(index=obs.index, dtype="object")
    binary[is_tumor] = "Tumor"
    binary[is_normal & ~is_tumor] = "Normal"  # safety: tumor wins ties

    keep_mask = binary.notna()
    print(f"[label] Total cells in input AnnData: {adata.n_obs}")
    print(f"[label] Cells matching Tumor rule    : {int(is_tumor.sum())}")
    print(f"[label] Cells matching Normal rule   : {int(is_normal.sum())}")
    print(f"[label] Total kept (Tumor or Normal) : {int(keep_mask.sum())}")
    print(f"[label] Cells dropped                : {int((~keep_mask).sum())}")

    sub = adata[keep_mask].copy()
    sub.obs["BinaryClass"] = binary[keep_mask].values
    sub.obs["Patient"] = sub.obs["Sample"].astype(str).map(extract_patient_id)
    return sub


def tokenize(adata: sc.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell: sort gene tokens by descending raw counts, truncate to MAX_GENES."""
    counts = adata.layers.get("counts", adata.X)
    if not sparse.issparse(counts):
        counts = sparse.csr_matrix(counts)
    else:
        counts = counts.tocsr()

    n_cells = adata.n_obs
    token_matrix = np.full((n_cells, MAX_GENES), fill_value=-1, dtype=np.int32)
    token_lengths = np.zeros(n_cells, dtype=np.int32)

    indptr = counts.indptr
    indices = counts.indices
    data = counts.data

    for cell_idx in trange(n_cells, desc="ranking"):
        start = indptr[cell_idx]
        end = indptr[cell_idx + 1]
        cell_gene_idx = indices[start:end]
        cell_expr = data[start:end]
        if cell_expr.size == 0:
            continue
        order = np.argsort(cell_expr)[::-1]  # descending by counts
        ranked = cell_gene_idx[order]
        if ranked.size > MAX_GENES:
            ranked = ranked[:MAX_GENES]
        token_matrix[cell_idx, : ranked.size] = ranked
        token_lengths[cell_idx] = ranked.size

    return token_matrix, token_lengths


def main() -> None:
    if not INPUT_H5AD.exists():
        raise FileNotFoundError(
            f"{INPUT_H5AD} not found. Run `python scripts/lung_qc_prep.py` first."
        )
    print(f"[load] {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)
    print(adata)

    print("\n[label] applying Cell_subtype-based labeling …")
    adata = apply_labels(adata)
    print(f"\n[label] BinaryClass distribution:\n{adata.obs['BinaryClass'].value_counts()}")
    print(f"\n[label] Patients per class:")
    print(adata.obs.groupby(["BinaryClass", "Patient"]).size().unstack(fill_value=0))

    print(f"\n[vocab] writing gene vocab ({adata.n_vars} genes) …")
    gene_vocab = pd.Series(
        data=np.arange(adata.n_vars, dtype=np.int32),
        index=adata.var_names,
        name="token_id",
    )
    gene_vocab.index.name = "gene_symbol"
    gene_vocab.to_csv(VOCAB_TSV, sep="\t", header=True)
    print(f"  → {VOCAB_TSV}")

    print("\n[tokenize] ranking genes per cell …")
    tokens, lengths = tokenize(adata)
    print(f"  tokens shape: {tokens.shape}")
    print(f"  length stats: min={lengths.min()} median={int(np.median(lengths))} "
          f"mean={int(lengths.mean())} max={lengths.max()}")

    print(f"\n[save] {TOKENS_NPZ}")
    np.savez_compressed(
        TOKENS_NPZ, tokens=tokens, lengths=lengths, max_genes=MAX_GENES
    )

    print(f"[save] {METADATA_TSV}")
    metadata = adata.obs[[
        "Patient", "Sample", "Sample_Origin",
        "Cell_type", "Cell_type.refined", "Cell_subtype",
        "BinaryClass",
    ]].copy()
    metadata["token_length"] = lengths
    metadata.to_csv(METADATA_TSV, sep="\t", index=False)

    print("\n[done] Tokenization complete.")
    print(f"  total cells     : {adata.n_obs}")
    print(f"  Tumor cells     : {int((metadata['BinaryClass'] == 'Tumor').sum())}")
    print(f"  Normal cells    : {int((metadata['BinaryClass'] == 'Normal').sum())}")
    print(f"  unique patients : {metadata['Patient'].nunique()}")


if __name__ == "__main__":
    main()
