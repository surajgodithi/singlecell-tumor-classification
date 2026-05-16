#!/usr/bin/env python
"""
Patient-level stratified train/val/test split for the lung dataset.

Reads `gse131907/processed/tokens/gse131907_tokens_metadata.tsv` (produced by
`scripts/lung_tokenize.py`) and produces `splits_by_patient.npz` containing
`train_idx`, `val_idx`, `test_idx` — cell-index arrays into the tokens matrix.

Splits are by **Patient ID** (extracted from `Sample` name) so no cell from
the same patient appears in two splits. Stratification ensures each split gets
both Tumor- and Normal-contributing patients in roughly the target proportion.

Default: 70 / 15 / 15.

Run:
    python scripts/lung_split.py                 # default
    python scripts/lung_split.py --seed 7
    python scripts/lung_split.py --train 0.6 --val 0.2 --test 0.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_DIR = Path("gse131907/processed/tokens")
META_TSV = TOKEN_DIR / "gse131907_tokens_metadata.tsv"
SPLITS_NPZ = TOKEN_DIR / "splits_by_patient.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument(
        "--min-cells-per-patient", type=int, default=20,
        help="Drop patients contributing fewer than this many kept cells (default: 20).",
    )
    return p.parse_args()


def categorize_patients(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-patient cell counts and which class(es) they contribute."""
    g = (
        meta.groupby(["Patient", "BinaryClass"])
        .size()
        .unstack(fill_value=0)
    )
    if "Tumor" not in g.columns:
        g["Tumor"] = 0
    if "Normal" not in g.columns:
        g["Normal"] = 0
    g["total"] = g["Tumor"] + g["Normal"]
    g["category"] = "both"
    g.loc[(g["Tumor"] > 0) & (g["Normal"] == 0), "category"] = "tumor_only"
    g.loc[(g["Tumor"] == 0) & (g["Normal"] > 0), "category"] = "normal_only"
    return g.reset_index()


def assign_split(
    patients_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """
    Stratified by category: assign each {tumor_only, normal_only, both} group
    to train/val/test independently in the target proportions, then concatenate.
    Guarantees each split has patients from every category that has ≥1 patient.
    """
    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for cat in ("tumor_only", "normal_only", "both"):
        pats = patients_df.loc[patients_df["category"] == cat, "Patient"].tolist()
        if not pats:
            continue
        pats_shuffled = list(pats)
        rng.shuffle(pats_shuffled)
        n = len(pats_shuffled)
        # Allocate at least 1 to val and test if there are ≥3 patients in the group
        n_val = max(1, round(n * val_frac)) if n >= 3 else (1 if n == 2 else 0)
        n_test = max(1, round(n * test_frac)) if n >= 3 else (1 if n == 1 else 0)
        # cap to leave at least 1 in train
        if n_val + n_test >= n:
            n_test = max(0, n - n_val - 1)
        n_train = n - n_val - n_test
        splits["train"].extend(pats_shuffled[:n_train])
        splits["val"].extend(pats_shuffled[n_train:n_train + n_val])
        splits["test"].extend(pats_shuffled[n_train + n_val:])
    return splits


def main() -> None:
    args = parse_args()
    fractions_sum = args.train + args.val + args.test
    if abs(fractions_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train+val+test must sum to 1.0, got {fractions_sum:.6f}"
        )
    if not META_TSV.exists():
        raise FileNotFoundError(
            f"{META_TSV} not found. Run `python scripts/lung_tokenize.py` first."
        )

    print(f"[load] {META_TSV}")
    meta = pd.read_csv(META_TSV, sep="\t")
    print(f"  total cells: {len(meta)}")
    print(f"  classes    : {meta['BinaryClass'].value_counts().to_dict()}")
    print(f"  patients   : {meta['Patient'].nunique()}")

    print("\n[stratify] patient categories:")
    pdf = categorize_patients(meta)
    print(pdf.groupby("category").size().to_string())

    # Drop tiny patients (noise filter)
    if args.min_cells_per_patient > 0:
        small = pdf[pdf["total"] < args.min_cells_per_patient]
        if not small.empty:
            print(f"\n[filter] dropping {len(small)} patients with <{args.min_cells_per_patient} cells:")
            print(small[["Patient", "Tumor", "Normal", "total"]].to_string(index=False))
            pdf = pdf[pdf["total"] >= args.min_cells_per_patient].copy()
            meta = meta[meta["Patient"].isin(pdf["Patient"])].copy()
            meta.reset_index(drop=True, inplace=True)

    print(f"\n[assign] split with seed={args.seed}")
    splits = assign_split(pdf, args.train, args.val, args.test, args.seed)

    for name, pats in splits.items():
        sub = meta[meta["Patient"].isin(pats)]
        cls = sub["BinaryClass"].value_counts().to_dict()
        print(f"  {name}: {len(pats)} patients, {len(sub):>6} cells, classes={cls}")
        print(f"         patients: {sorted(pats)}")

    # Convert to cell index arrays
    train_idx = meta.index[meta["Patient"].isin(splits["train"])].to_numpy(dtype=np.int64)
    val_idx = meta.index[meta["Patient"].isin(splits["val"])].to_numpy(dtype=np.int64)
    test_idx = meta.index[meta["Patient"].isin(splits["test"])].to_numpy(dtype=np.int64)

    # Sanity: no overlap
    overlap_tv = set(train_idx) & set(val_idx)
    overlap_tt = set(train_idx) & set(test_idx)
    overlap_vt = set(val_idx) & set(test_idx)
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"split overlap: train∩val={len(overlap_tv)} train∩test={len(overlap_tt)} "
            f"val∩test={len(overlap_vt)}"
        )

    print(f"\n[save] {SPLITS_NPZ}")
    np.savez(
        SPLITS_NPZ,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_patients=np.array(splits["train"], dtype=object),
        val_patients=np.array(splits["val"], dtype=object),
        test_patients=np.array(splits["test"], dtype=object),
    )
    print("[done]")


if __name__ == "__main__":
    main()
