#!/usr/bin/env python
"""
Lung dataset (GSE131907) download + QC prep.

Mirrors `notebooks/lung_01_quality_control.ipynb` but as a runnable script
(Colab-specific cells stripped, save step added). Writes the filtered AnnData
to `gse131907/processed/gse131907_filtered_raw.h5ad`, which `lung_02_*` expects.

Run:
    python scripts/lung_qc_prep.py

Steps:
1. Download raw counts + annotation from NCBI GEO (~15 GB compressed counts).
2. Chunk-load counts into a sparse CSR matrix to stay memory-safe.
3. Build AnnData, attach annotation, set Class=Sample_Origin and Patient=Sample.
4. Standard QC filtering: min 200 genes/cell, gene must appear in min 3 cells.
5. Normalize total + log1p, compute HVGs (top 5000).
6. Save filtered AnnData and print per-patient × per-class summary so we can
   decide the binary-collapse strategy before tokenisation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import scanpy as sc
from scipy import sparse


DATA_DIR = Path("gse131907")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_H5AD = PROCESSED_DIR / "gse131907_filtered_raw.h5ad"

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/"
FILES = {
    "counts": "GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz",
    "annotation": "GSE131907_Lung_Cancer_cell_annotation.txt.gz",
}


def download() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, filename in FILES.items():
        url = f"{BASE_URL}{filename}"
        path = pooch.retrieve(
            url=url, known_hash=None, fname=filename, path=RAW_DIR, progressbar=True
        )
        paths[label] = Path(path)
        size_mb = paths[label].stat().st_size / 1e6
        print(f"[downloaded] {label}: {path} ({size_mb:.0f} MB)")
    return paths


def load_counts_chunked(counts_path: Path) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Chunk-read the gene-row × cell-column TSV into a sparse (cells × genes) CSR.

    The file is gene-row × cell-column with ~208k cells per row. At `chunksize=2000`
    each chunk is 2000 × 208k = 400 M values (~1.6 GB) which OOMs / hangs on 32 GB.
    Use a much smaller chunksize so each pandas chunk is ~10-30 MB; processing is
    slower in chunks but memory stays bounded and we get progress visibility.
    """
    CHUNK_ROWS = 100  # ~100 × 208k × 4B = ~83 MB per chunk peak

    blocks: list[sparse.csr_matrix] = []
    gene_names: list[str] = []
    cell_names: list[str] | None = None
    n_rows_seen = 0
    print(f"  chunksize={CHUNK_ROWS} rows; expecting ~{24_000 // CHUNK_ROWS} chunks")
    for chunk in pd.read_csv(
        counts_path, sep="\t", index_col=0, chunksize=CHUNK_ROWS,
        low_memory=False,  # dtype inferred — applying int32 to index col fails on gene names
    ):
        if cell_names is None:
            cell_names = chunk.columns.tolist()
        # Convert dense chunk → sparse immediately; the dense slice is GC'd next iter.
        # Cast values to int32 manually so the resulting sparse blocks use 4 bytes/value.
        blocks.append(sparse.csr_matrix(chunk.to_numpy(dtype=np.int32, copy=False)))
        gene_names.extend(chunk.index.tolist())
        n_rows_seen += len(chunk)
        if n_rows_seen % 1000 == 0:
            print(f"  loaded {n_rows_seen} genes (total blocks: {len(blocks)})", flush=True)
    print(f"  loaded {n_rows_seen} genes total. vstacking {len(blocks)} blocks …", flush=True)
    counts_csr = sparse.vstack(blocks)  # genes x cells
    print(f"  vstack done. transposing to cells x genes …", flush=True)
    counts_csr = counts_csr.T.tocsr()   # cells x genes
    assert cell_names is not None
    return counts_csr, cell_names, gene_names


def main() -> None:
    print(f"Raw dir       : {RAW_DIR}")
    print(f"Processed dir : {PROCESSED_DIR}")
    print(f"Output AnnData: {OUT_H5AD}")
    print()

    paths = download()

    print("\n[load] annotation …")
    meta_df = pd.read_csv(paths["annotation"], sep="\t", index_col=0)
    print(f"  annotation rows: {len(meta_df)}")
    print(f"  annotation columns: {list(meta_df.columns)}")

    print("\n[load] counts matrix (chunked) …")
    counts_csr, cell_names, gene_names = load_counts_chunked(paths["counts"])
    print(f"  shape: cells={counts_csr.shape[0]}  genes={counts_csr.shape[1]}")
    print(f"  nnz  : {counts_csr.nnz:,}")

    print("\n[build] AnnData …")
    adata = sc.AnnData(X=counts_csr)
    adata.obs_names = cell_names
    adata.var_names = gene_names
    adata.obs = meta_df.reindex(cell_names)
    adata.obs["Class"] = adata.obs["Sample_Origin"].astype(str)
    adata.obs["Patient"] = adata.obs["Sample"].astype(str)
    adata.layers["counts"] = adata.X.copy()
    print(adata)

    print("\n[qc] calculate metrics …")
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    print(f"  n_genes_by_counts  median={adata.obs['n_genes_by_counts'].median():.0f}")
    print(f"  total_counts       median={adata.obs['total_counts'].median():.0f}")

    print("\n[qc] filter cells/genes (min_genes=200, min_cells=3) …")
    adata = adata[adata.obs["n_genes_by_counts"] >= 200].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  after filter: cells={adata.n_obs}  genes={adata.n_vars}")

    print("\n[norm] normalize_total + log1p + HVG (top 5000) …")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    # restore counts as the .X for downstream tokenization
    adata.X = adata.layers["counts"].copy()
    sc.pp.highly_variable_genes(
        adata, flavor="cell_ranger", n_top_genes=5000, subset=False, layer="log1p_norm"
    )
    print(adata.var["highly_variable"].value_counts())

    print("\n[summary] cells per Sample_Origin (Class):")
    print(adata.obs["Class"].value_counts())

    print("\n[summary] cells per Patient × Class (Sample_Origin):")
    pivot = (
        adata.obs.groupby(["Patient", "Class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print(pivot)
    print(f"\nTotal patients: {pivot.shape[0]}")

    print(f"\n[save] writing AnnData → {OUT_H5AD}")
    adata.write(OUT_H5AD)
    size_mb = OUT_H5AD.stat().st_size / 1e6
    print(f"  done. file size: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
