#!/usr/bin/env python
"""
Cross-donor aggregation of in silico perturbation results.

Reads per-fold perturbation TSVs produced by in_silico_perturbation.py:
  results/lodo/fold_{donor}_perturbation.tsv

Per (gene, class) pair, aggregates across available folds:
  - n_folds_present       : number of folds that contain this gene+class
  - mean_delta            : mean delta (P(Tumor)|perturbed − baseline) across folds
  - std_delta             : standard deviation of delta across folds
  - mean_flip_fraction    : mean flip_fraction across folds
  - n_donors_ranked_top50 : folds in which gene ranks in top-50 by |delta|
  - consistency_score     : fraction of folds where delta shares the sign of mean_delta

Output: results/perturbation_aggregate.tsv

Biology key:
  Tumor cells  — large negative mean_delta → oncogene candidate (removal kills Tumor signal)
  Normal cells — large positive mean_delta → TSG candidate (removal gains Tumor signal)

Usage
-----
    python scripts/aggregate_perturbation.py
    python scripts/aggregate_perturbation.py --results-dir results/lodo --output results/perturbation_aggregate.tsv
    python scripts/aggregate_perturbation.py --class Tumor   # only Tumor rows
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALL_DONORS = ["KUL01", "KUL19", "KUL21", "KUL28", "KUL30", "KUL31"]
REQUIRED_COLS = {"gene", "class", "delta"}
OUTPUT_COLS = [
    "gene", "class", "phase",
    "n_folds_present",
    "mean_delta", "std_delta",
    "mean_flip_fraction",
    "n_donors_ranked_top50",
    "consistency_score",
    "folds_present",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir", type=Path, default=Path("results/lodo"),
        help="Directory containing fold_*_perturbation.tsv files (default: results/lodo).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/perturbation_aggregate.tsv"),
        help="Output TSV path (default: results/perturbation_aggregate.tsv).",
    )
    p.add_argument(
        "--donors", nargs="+", metavar="DONOR", default=None,
        help="Subset of donors to aggregate (default: all available).",
    )
    p.add_argument(
        "--class", dest="cls_filter", default=None,
        choices=["Tumor", "Normal"],
        help="Restrict output to one cell class.",
    )
    p.add_argument(
        "--top-n", type=int, default=50,
        help="n for n_donors_ranked_topN metric (default: 50).",
    )
    return p.parse_args()


def load_fold_tsv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as e:
        print(f"  [warn] Could not load {path}: {e}")
        return None

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"  [warn] {path.name} missing columns {missing} — skipping.")
        return None
    return df


def aggregate(
    frames: list[tuple[str, pd.DataFrame]],
    top_n: int,
) -> pd.DataFrame:
    """
    frames : list of (donor, DataFrame) pairs
    Returns aggregated DataFrame per (gene, class).
    """
    combined = pd.concat(
        [df.assign(fold_donor=donor) for donor, df in frames],
        ignore_index=True,
    )

    # Pre-compute top-N gene sets per (class, fold_donor) — used for n_donors_ranked_top50
    top_n_sets: dict[tuple[str, str], set[str]] = {}
    for (cls, donor), fold_grp in combined.groupby(["class", "fold_donor"]):
        top_genes = (
            fold_grp.assign(abs_delta=fold_grp["delta"].abs())
            .sort_values("abs_delta", ascending=False)
            .head(top_n)["gene"]
            .tolist()
        )
        top_n_sets[(cls, donor)] = set(top_genes)

    # Determine phase per (gene, class): "known" if any fold marked it known, else "discovery"
    phase_df = (
        combined.groupby(["gene", "class"])["phase"]
        .apply(lambda s: "known" if "known" in s.values else "discovery")
        .reset_index()
    ) if "phase" in combined.columns else None

    all_donors = sorted(combined["fold_donor"].unique().tolist())

    rows = []
    for (gene, cls), grp in combined.groupby(["gene", "class"]):
        deltas = grp["delta"].values
        n_folds = len(deltas)

        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas)) if n_folds > 1 else float("nan")

        # flip_fraction is optional (may be absent in old TSVs)
        if "flip_fraction" in grp.columns and grp["flip_fraction"].notna().any():
            mean_flip = float(np.nanmean(grp["flip_fraction"].values))
        else:
            mean_flip = float("nan")

        # n_donors_ranked_top50: count folds where this gene is in the top-N by |delta|
        n_top = sum(
            1 for donor in all_donors
            if gene in top_n_sets.get((cls, donor), set())
        )

        # consistency_score: fraction of folds where sign(delta) == sign(mean_delta)
        if mean_d == 0.0:
            consistency = float("nan")
        else:
            same_sign = np.sign(deltas) == np.sign(mean_d)
            consistency = float(same_sign.sum() / n_folds)

        folds_present = sorted(grp["fold_donor"].unique().tolist())

        rows.append({
            "gene": gene,
            "class": cls,
            "n_folds_present": n_folds,
            "mean_delta": mean_d,
            "std_delta": std_d,
            "mean_flip_fraction": mean_flip,
            "n_donors_ranked_top50": n_top,
            "consistency_score": consistency,
            "folds_present": ",".join(folds_present),
        })

    result = pd.DataFrame(rows)

    if phase_df is not None:
        result = result.merge(phase_df, on=["gene", "class"], how="left")
    else:
        result["phase"] = "unknown"

    return result


def sort_for_biology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tumor rows: most negative mean_delta first (oncogene candidates).
    Normal rows: most positive mean_delta first (TSG candidates).
    Returns combined DataFrame with Tumor rows first.
    """
    tumor = df[df["class"] == "Tumor"].sort_values("mean_delta", ascending=True)
    normal = df[df["class"] == "Normal"].sort_values("mean_delta", ascending=False)
    return pd.concat([tumor, normal], ignore_index=True)


def main() -> None:
    args = parse_args()
    donors = args.donors or ALL_DONORS

    print(f"Results dir : {args.results_dir}")
    print(f"Donors      : {donors}")
    print(f"Top-N       : {args.top_n}")

    frames: list[tuple[str, pd.DataFrame]] = []
    for donor in donors:
        tsv = args.results_dir / f"fold_{donor}_perturbation.tsv"
        if not tsv.exists():
            print(f"  [info] {tsv.name} not found — skipping.")
            continue
        df = load_fold_tsv(tsv)
        if df is not None:
            frames.append((donor, df))
            print(f"  Loaded {tsv.name}: {len(df)} rows")

    if not frames:
        print("No perturbation TSVs found. Run in_silico_perturbation.py first.")
        return

    print(f"\nAggregating {len(frames)} fold(s) …")
    result = aggregate(frames, top_n=args.top_n)

    if args.cls_filter:
        result = result[result["class"] == args.cls_filter].copy()

    result = sort_for_biology(result)

    # Keep only columns present (phase may be absent in old TSV format)
    final_cols = [c for c in OUTPUT_COLS if c in result.columns]
    result = result[final_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved → {args.output}  ({len(result)} rows)")

    # Print top 10 Tumor and top 10 Normal candidates
    for cls, sign_label in [("Tumor", "most negative delta"), ("Normal", "most positive delta")]:
        sub = result[result["class"] == cls].head(10)
        if sub.empty:
            continue
        print(f"\nTop 10 {cls} candidates ({sign_label}):")
        print(
            sub[["gene", "mean_delta", "n_donors_ranked_top50", "consistency_score",
                 "n_folds_present"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
