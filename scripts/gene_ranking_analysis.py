#!/usr/bin/env python
"""
Gene Ranking Analysis for LODO folds.

Part A — Raw expression ranking:
    For the held-out donor, compute mean expression per gene separately for
    Tumor and Normal cells and produce a dot plot of the top 50 genes.

Part B — Attention weight ranking:
    Load the fold's trained checkpoint, run inference with output_attentions=True,
    extract attention from the final transformer layer, and rank genes by mean
    attention weight received across cells of each class.

Part C — Combined figure:
    Side-by-side expression (left) and attention (right) dot plots for both classes.

Usage
-----
# Test without a trained checkpoint (uses base Geneformer weights — random attention):
    python scripts/gene_ranking_analysis.py --donors KUL01 --use-base-model

# Run for a specific fold (requires outputs/lodo/fold_KUL01/ checkpoint):
    python scripts/gene_ranking_analysis.py --donors KUL01

# All folds:
    python scripts/gene_ranking_analysis.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent))
from finetune_transformer import (
    RankedGeneCollator,
    RankedGeneDataset,
    build_vocab_remap,
    ensure_label_column,
    find_single,
    load_gene_name_dict,
    load_gene_vocab,
    prepare_label_mappings,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Path("configs/lung_perturbation.yaml")
TOKEN_GLOB = "*_gene_rank_tokens.npz"
META_GLOB = "*_tokens_metadata.tsv"
ALL_DONORS = ["KUL01", "KUL19", "KUL21", "KUL28", "KUL30", "KUL31"]
ANNDATA_PATH = Path("gse144735/processed/gse144735_filtered_raw.h5ad")
CRC_MARKERS = ["KRAS", "APC", "TP53", "MYC", "EGFR", "BRAF", "EPCAM", "CDX2", "SMAD4", "PIK3CA"]
LUNG_ADC_MARKERS = ["EGFR", "KRAS", "ALK", "TP53", "STK11", "KEAP1", "NF1", "RBM10", "MET", "ERBB2"]
# Resolved at runtime in main() from cfg["marker_panel"]; defaults to LUAD panel.
MARKERS: list = list(LUNG_ADC_MARKERS)
MARKER_LABEL: str = "LUAD marker"
TOP_N = 50
TUMOR_COLOR = "#e74c3c"
NORMAL_COLOR = "#3498db"
MARKER_COLOR = "#f39c12"


def _resolve_marker_panel(panel: Optional[str]) -> Tuple[list, str]:
    """Return (markers, legend_label) for a config marker_panel value."""
    p = (panel or "lung_adc").lower()
    if p == "lung_adc":
        return list(LUNG_ADC_MARKERS), "LUAD marker"
    if p == "crc":
        return list(CRC_MARKERS), "CRC marker"
    if p == "both":
        return list(dict.fromkeys(LUNG_ADC_MARKERS + CRC_MARKERS)), "Driver marker"
    if p == "none":
        return [], "Marker"
    raise ValueError(f"Unknown marker_panel '{panel}'. Use lung_adc | crc | both | none.")
FALLBACK_DEFAULTS: dict = {
    "tokens_dir": "gse144735/processed/tokens",
    "output_dir": "outputs/lodo",
    "results_dir": "results/lodo",
    "anndata_path": "gse144735/processed/gse144735_filtered_raw.h5ad",
    "model_name_or_path": "Geneformer/Geneformer-V2-104M",
    "model_vocab": "Geneformer/geneformer/token_dictionary_gc104M.pkl",
    "model_gene_name_dict": "Geneformer/geneformer/gene_name_id_dict_gc104M.pkl",
    "label_column": "BinaryClass",
    "max_length": 2048,
    "seed": 42,
    "attn_batch_size": 2,
}


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--donors", nargs="+", metavar="DONOR", default=None,
                   help="LODO donors to analyse (CRC mode). Ignored if --single-fold.")
    p.add_argument("--single-fold", action="store_true",
                   help="Run on a single train/val/test split (lung mode). Uses test_idx from "
                        "splits_by_patient.npz; one set of outputs named by `output_prefix`.")
    p.add_argument("--split-set", choices=["test", "val"], default="test",
                   help="In --single-fold mode, which split to rank on (default: test).")
    p.add_argument("--output-prefix", type=str, default=None,
                   help="In --single-fold mode, output file prefix (default: config 'output_prefix' "
                        "or 'lung_test').")
    p.add_argument("--use-base-model", action="store_true",
                   help="Use base Geneformer weights instead of fold checkpoint. "
                        "Attention weights will be untrained (for pipeline testing).")
    p.add_argument("--skip-attention", action="store_true",
                   help="Run Part A only (no model loading required).")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cfg = {}
    if path and path.exists():
        data = yaml.safe_load(path.read_text())
        cfg = data if isinstance(data, dict) else {}
        print(f"Loaded config: {path}")
    for k, v in FALLBACK_DEFAULTS.items():
        cfg.setdefault(k, v)
    for field in ("tokens_dir", "output_dir", "results_dir", "model_vocab", "model_gene_name_dict"):
        if cfg.get(field) is not None:
            cfg[field] = Path(cfg[field])
    return cfg


# ---------------------------------------------------------------------------
# Shared vocab / remap helpers
# ---------------------------------------------------------------------------

def build_model_token_to_gene(
    dataset_vocab: pd.Series,
    remap: np.ndarray,
    model_pad_token_id: int,
) -> Dict[int, str]:
    """Map model token IDs back to gene symbols for attention interpretation."""
    mapping: Dict[int, str] = {}
    for gene_sym, dataset_id in dataset_vocab.items():
        model_id = int(remap[int(dataset_id)])
        if model_id != model_pad_token_id:
            mapping[model_id] = str(gene_sym)
    return mapping


# ---------------------------------------------------------------------------
# Part A — raw expression ranking
# ---------------------------------------------------------------------------

def compute_expression_ranking(
    adata: sc.AnnData,
    patient_ids: list,
    label_column: str,
    label: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict with keys "Tumor" and "Normal", each a DataFrame:
      gene | mean_expr | rank | is_marker
    ranked descending by mean_expr, top TOP_N rows only.

    `patient_ids` is a list of patient IDs (as strings); cells where
    metadata.Patient ∈ patient_ids will be included. `label` is used only
    for log messages.
    """
    pat_str = [str(p) for p in patient_ids]
    donor_mask = adata.obs["Patient"].astype(str).isin(pat_str)
    results = {}
    for cls in ("Tumor", "Normal"):
        cls_mask = donor_mask & (adata.obs[label_column] == cls)
        n_cells = cls_mask.sum()
        if n_cells == 0:
            print(f"  [warn] No {cls} cells found for {label or pat_str} — skipping.")
            continue

        # Use counts layer (raw UMI)
        if "counts" in adata.layers:
            X = adata.layers["counts"][cls_mask]
        else:
            X = adata.X[cls_mask]

        # Mean expression per gene (dense computation; 24k genes × ~few k cells is fine)
        if hasattr(X, "toarray"):
            mean_expr = np.asarray(X.mean(axis=0)).ravel()
        else:
            mean_expr = np.asarray(X.mean(axis=0)).ravel()

        gene_names = list(adata.var_names)
        df = pd.DataFrame({"gene": gene_names, "mean_expr": mean_expr})
        df = df.sort_values("mean_expr", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["is_marker"] = df["gene"].isin(MARKERS)
        results[cls] = df.head(TOP_N).copy()
        print(f"  [{label or ','.join(pat_str)}] {cls}: {n_cells} cells, top gene = {df.iloc[0]['gene']} "
              f"(mean={df.iloc[0]['mean_expr']:.2f})")
    return results


def _dot_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    point_color: str,
    y_label: str,
) -> None:
    """Shared dot-plot renderer for Parts A and B."""
    markers_in_top = df[df["is_marker"]]
    non_markers = df[~df["is_marker"]]

    ax.scatter(non_markers["rank"], non_markers[y_col],
               color=point_color, s=30, alpha=0.7, zorder=2, label="Gene")
    ax.scatter(markers_in_top["rank"], markers_in_top[y_col],
               color=MARKER_COLOR, s=80, marker="*", zorder=3, label=MARKER_LABEL)

    for _, row in markers_in_top.iterrows():
        ax.annotate(row["gene"], (row["rank"], row[y_col]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color=MARKER_COLOR, fontweight="bold")

    ax.set_xlabel("Rank", fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_expression_ranking(
    rankings: Dict[str, pd.DataFrame],
    label: str,
    output_prefix: str,
    results_dir: Path,
) -> None:
    n_classes = len(rankings)
    if n_classes == 0:
        return

    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes), squeeze=False)
    class_colors = {"Tumor": TUMOR_COLOR, "Normal": NORMAL_COLOR}

    for i, (cls, df) in enumerate(rankings.items()):
        _dot_plot(
            axes[i, 0], df, "mean_expr",
            title=f"{label} — {cls} — Top {TOP_N} by Mean Expression",
            point_color=class_colors.get(cls, "#666666"),
            y_label="Mean raw count",
        )

    fig.suptitle(f"Expression Ranking — {label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = results_dir / f"{output_prefix}_expression_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Part A.5 — log2 fold-change ranking (differential expression)
# ---------------------------------------------------------------------------
# Raw expression ranking is dominated by housekeeping / lineage genes that are
# abundant in both classes. logFC asks "what *changes* between Tumor and Normal"
# regardless of absolute level — surfaces modestly-expressed but discriminative
# genes (the classical DEA view), complementing attention ranking.

def compute_logfc_ranking(
    adata: sc.AnnData,
    patient_ids: list,
    label_column: str,
    label: str = "",
) -> Dict[str, pd.DataFrame]:
    """Per class, return the FULL gene table ranked by descending log2 fold
    change of that class vs the rest, with Wilcoxon p-values. Plotting slices
    head(TOP_N); save_logfc_tsv writes the whole thing for downstream use."""
    pat_str = [str(p) for p in patient_ids]
    donor_mask = adata.obs["Patient"].astype(str).isin(pat_str)
    sub = adata[donor_mask].copy()
    classes_present = set(sub.obs[label_column].dropna().unique())
    if "Tumor" not in classes_present or "Normal" not in classes_present:
        print(f"  [warn] logFC needs both Tumor and Normal; "
              f"{label} has {sorted(classes_present)} — skipping.")
        return {}

    # Use log1p-normalized layer if available so log2FC is meaningful.
    layer = "log1p_norm" if "log1p_norm" in sub.layers else None
    sub.obs[label_column] = sub.obs[label_column].astype("category")
    sc.tl.rank_genes_groups(
        sub, groupby=label_column, method="wilcoxon",
        layer=layer, use_raw=False,
    )

    results = {}
    for cls in ("Tumor", "Normal"):
        if cls not in sub.obs[label_column].unique():
            continue
        df = sc.get.rank_genes_groups_df(sub, group=cls)
        df = df.rename(columns={"names": "gene", "logfoldchanges": "log2fc"})
        df = df.sort_values("log2fc", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["is_marker"] = df["gene"].isin(MARKERS)
        results[cls] = df  # full table; consumers slice as needed
        top = df.iloc[0]
        print(f"  [{label or ','.join(pat_str)}] {cls} logFC: top gene = {top['gene']} "
              f"(log2FC={top['log2fc']:.2f}, p_adj={top['pvals_adj']:.2e})")
    return results


def plot_logfc_ranking(
    rankings: Dict[str, pd.DataFrame],
    label: str,
    output_prefix: str,
    results_dir: Path,
) -> None:
    n_classes = len(rankings)
    if n_classes == 0:
        return
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes), squeeze=False)
    class_colors = {"Tumor": TUMOR_COLOR, "Normal": NORMAL_COLOR}
    for i, (cls, df) in enumerate(rankings.items()):
        _dot_plot(
            axes[i, 0], df.head(TOP_N).copy(), "log2fc",
            title=f"{label} — {cls} — Top {TOP_N} by log2 Fold Change vs other class",
            point_color=class_colors.get(cls, "#666666"),
            y_label="log2 fold change",
        )
    fig.suptitle(f"Differential Expression (logFC) Ranking — {label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = results_dir / f"{output_prefix}_logfc_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_logfc_tsv(
    rankings: Dict[str, pd.DataFrame],
    output_prefix: str,
    results_dir: Path,
) -> None:
    """Write a per-gene TSV with log2fc_tumor, log2fc_normal, p-values."""
    rows = []
    for cls, df in rankings.items():
        # Keep full ranking from scanpy, not just top TOP_N, for downstream use.
        # df here is the head(TOP_N); we need full data for the TSV.
        tmp = df[["rank", "gene", "log2fc", "pvals_adj", "is_marker"]].copy()
        tmp.insert(0, "class", cls)
        rows.append(tmp)
    if not rows:
        return
    combined = pd.concat(rows, ignore_index=True)
    pivot_fc = combined.pivot_table(
        index="gene", columns="class", values="log2fc", aggfunc="first"
    ).reset_index()
    pivot_fc.columns.name = None
    pivot_p = combined.pivot_table(
        index="gene", columns="class", values="pvals_adj", aggfunc="first"
    )
    for cls in ("Tumor", "Normal"):
        if cls not in pivot_fc.columns:
            pivot_fc[cls] = float("nan")
        if cls not in pivot_p.columns:
            pivot_p[cls] = float("nan")
    pivot_fc = pivot_fc.rename(columns={"Tumor": "log2fc_tumor", "Normal": "log2fc_normal"})
    pivot_fc["padj_tumor"] = pivot_fc["gene"].map(pivot_p["Tumor"])
    pivot_fc["padj_normal"] = pivot_fc["gene"].map(pivot_p["Normal"])
    pivot_fc["is_marker"] = pivot_fc["gene"].isin(MARKERS)
    pivot_fc = pivot_fc.sort_values("log2fc_tumor", ascending=False, na_position="last")
    pivot_fc["rank_tumor"] = range(1, len(pivot_fc) + 1)
    out = results_dir / f"{output_prefix}_logfc_genes.tsv"
    pivot_fc.to_csv(out, sep="\t", index=False)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Part B — attention weight ranking
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_fold_model(
    fold_output_dir: Path,
    use_base_model: bool,
    base_model_path: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    device: torch.device,
) -> Optional[AutoModelForSequenceClassification]:
    if use_base_model:
        model_path = base_model_path
        print(f"  [attention] Using base model weights (pipeline test).")
    else:
        ckpt_file = fold_output_dir / "best_checkpoint.txt"
        if not ckpt_file.exists():
            print(f"  [warn] No checkpoint found at {fold_output_dir}. "
                  "Run lodo_cv.py first, or pass --use-base-model for testing.")
            return None
        model_path = ckpt_file.read_text().strip()
        print(f"  [attention] Loading checkpoint: {model_path}")

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    # eager attention is required for output_attentions=True (sdpa does not support it)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, attn_implementation="eager"
    )
    model.eval()
    model.to(device)
    return model


def extract_attention_rankings(
    tokens: np.ndarray,
    metadata: pd.DataFrame,
    encoded_labels: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    remap: np.ndarray,
    pad_fill_value: int,
    model: AutoModelForSequenceClassification,
    model_token_to_gene: Dict[int, str],
    patient_ids: list,
    label: str,
    cfg: dict,
    device: torch.device,
) -> Dict[str, pd.DataFrame]:
    """
    For each class (Tumor, Normal), run inference on cells from `patient_ids` with
    output_attentions=True. Accumulate mean attention received per gene token across
    all cells in the class, across all heads of the final transformer layer.

    `label` is used purely for log messages (e.g. "KUL01" or "lung_test").

    Returns dict {"Tumor": ranked DataFrame, "Normal": ranked DataFrame}.
    """
    results = {}
    num_labels = len(id2label)
    max_length = int(cfg["max_length"])
    attn_batch_size = int(cfg.get("attn_batch_size", 2))
    pat_str = [str(p) for p in patient_ids]
    in_split = metadata["Patient"].astype(str).isin(pat_str)

    for cls in ("Tumor", "Normal"):
        cls_mask = in_split & (metadata[cfg["label_column"]] == cls)
        cls_idx = np.where(cls_mask)[0].astype(np.int64)
        if len(cls_idx) == 0:
            print(f"  [warn] No {cls} cells for {label}.")
            continue

        print(f"  [{label}] Extracting attention for {cls} ({len(cls_idx)} cells, "
              f"batch_size={attn_batch_size}) …")

        ds = RankedGeneDataset(
            tokens, cls_idx, encoded_labels, pad_fill_value,
            max_length, remap,
            token_mask_prob=0.0, mixup_prob=0.0, num_labels=num_labels,
        )
        collator = RankedGeneCollator(num_labels=num_labels)
        loader = DataLoader(
            ds, batch_size=attn_batch_size, collate_fn=collator,
            shuffle=False, num_workers=0, pin_memory=False,
        )

        # Accumulate: model_token_id → total attention received, count of appearances
        token_attn_sum: dict[int, float] = {}
        token_count: dict[int, int] = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)       # (B, L)
                attn_mask = batch["attention_mask"].to(device)  # (B, L)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_attentions=True,
                )

                # Last transformer layer: (B, H, L, L)
                last_attn = outputs.attentions[-1].float()

                # Average over heads → (B, L, L)
                # [b, i, j] = mean attention from query-position i to key-position j
                last_attn = last_attn.mean(dim=1)

                # Mean attention RECEIVED by each key position
                # (average over all query positions → dim 1):
                # result[b, j] = mean_i attn[b, i, j]
                attn_received = last_attn.mean(dim=1)  # (B, L)

                # Zero out padding positions
                attn_received = attn_received * attn_mask.float()

                attn_received_cpu = attn_received.cpu().numpy()
                input_ids_cpu = input_ids.cpu().numpy()
                attn_mask_cpu = attn_mask.cpu().numpy()

                for b in range(input_ids_cpu.shape[0]):
                    seq_len = int(attn_mask_cpu[b].sum())
                    for pos in range(seq_len):
                        tok_id = int(input_ids_cpu[b, pos])
                        val = float(attn_received_cpu[b, pos])
                        token_attn_sum[tok_id] = token_attn_sum.get(tok_id, 0.0) + val
                        token_count[tok_id] = token_count.get(tok_id, 0) + 1

                if batch_idx % 50 == 0:
                    print(f"    batch {batch_idx}/{len(loader)}", end="\r")

        print()

        # Build ranked DataFrame
        rows = []
        for tok_id, total in token_attn_sum.items():
            gene = model_token_to_gene.get(tok_id)
            if gene is None:
                continue
            mean_attn = total / token_count[tok_id]
            rows.append({"gene": gene, "mean_attn": mean_attn})

        if not rows:
            print(f"  [warn] No gene-mapped tokens found for {cls} {label}.")
            continue

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_attn", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["is_marker"] = df["gene"].isin(MARKERS)
        results[cls] = df
        print(f"  [{label}] {cls} attention: top gene = {df.iloc[0]['gene']} "
              f"(mean_attn={df.iloc[0]['mean_attn']:.5f})")

    return results


def plot_attention_ranking(
    rankings: Dict[str, pd.DataFrame],
    label: str,
    output_prefix: str,
    results_dir: Path,
) -> None:
    n_classes = len(rankings)
    if n_classes == 0:
        return

    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 5 * n_classes), squeeze=False)
    class_colors = {"Tumor": TUMOR_COLOR, "Normal": NORMAL_COLOR}

    for i, (cls, df) in enumerate(rankings.items()):
        top50 = df.head(TOP_N).copy()
        _dot_plot(
            axes[i, 0], top50, "mean_attn",
            title=f"{label} — {cls} — Top {TOP_N} by Attention Weight",
            point_color=class_colors.get(cls, "#666666"),
            y_label="Mean attention received",
        )

    fig.suptitle(f"Attention Ranking — {label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = results_dir / f"{output_prefix}_attention_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_attention_tsv(
    rankings: Dict[str, pd.DataFrame],
    output_prefix: str,
    results_dir: Path,
) -> None:
    rows = []
    for cls, df in rankings.items():
        tmp = df[["rank", "gene", "mean_attn", "is_marker"]].copy()
        tmp.insert(0, "class", cls)
        rows.append(tmp)
    if not rows:
        return
    combined = pd.concat(rows, ignore_index=True)
    # Pivot so each gene has one row with separate Tumor/Normal columns
    pivot = combined.pivot_table(
        index="gene", columns="class", values="mean_attn", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    for cls in ("Tumor", "Normal"):
        if cls not in pivot.columns:
            pivot[cls] = float("nan")
    pivot = pivot.rename(columns={"Tumor": "mean_attn_tumor", "Normal": "mean_attn_normal"})
    pivot["is_marker"] = pivot["gene"].isin(MARKERS)
    # Rank by tumor attention for the output ordering
    if "mean_attn_tumor" in pivot.columns:
        pivot = pivot.sort_values("mean_attn_tumor", ascending=False, na_position="last")
    pivot["rank_tumor"] = range(1, len(pivot) + 1)
    out = results_dir / f"{output_prefix}_attention_genes.tsv"
    pivot.to_csv(out, sep="\t", index=False)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Part C — combined figure
# ---------------------------------------------------------------------------

def plot_combined(
    expr_rankings: Dict[str, pd.DataFrame],
    attn_rankings: Dict[str, pd.DataFrame],
    label: str,
    output_prefix: str,
    results_dir: Path,
    logfc_rankings: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    logfc_rankings = logfc_rankings or {}
    classes = [
        c for c in ("Tumor", "Normal")
        if c in expr_rankings or c in attn_rankings or c in logfc_rankings
    ]
    if not classes:
        return

    n_rows = len(classes)
    fig, axes = plt.subplots(n_rows, 3, figsize=(27, 5 * n_rows), squeeze=False)
    class_colors = {"Tumor": TUMOR_COLOR, "Normal": NORMAL_COLOR}

    for row, cls in enumerate(classes):
        color = class_colors.get(cls, "#666666")

        if cls in expr_rankings:
            _dot_plot(
                axes[row, 0],
                expr_rankings[cls].head(TOP_N),
                "mean_expr",
                title=f"{cls} — Expression Ranking",
                point_color=color,
                y_label="Mean raw count",
            )
        else:
            axes[row, 0].text(0.5, 0.5, "No data", ha="center", va="center")

        if cls in logfc_rankings:
            _dot_plot(
                axes[row, 1],
                logfc_rankings[cls].head(TOP_N),
                "log2fc",
                title=f"{cls} — log2 Fold Change Ranking",
                point_color=color,
                y_label="log2 fold change",
            )
        else:
            axes[row, 1].text(0.5, 0.5, "No logFC", ha="center", va="center")

        if cls in attn_rankings:
            _dot_plot(
                axes[row, 2],
                attn_rankings[cls].head(TOP_N),
                "mean_attn",
                title=f"{cls} — Attention Ranking",
                point_color=color,
                y_label="Mean attention received",
            )
        else:
            axes[row, 2].text(0.5, 0.5, "No checkpoint", ha="center", va="center")

    fig.suptitle(
        f"Expression vs logFC vs Attention Ranking — {label}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = results_dir / f"{output_prefix}_combined_ranking.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # Resolve which marker panel highlights stars + labels the legend on plots.
    global MARKERS, MARKER_LABEL
    MARKERS, MARKER_LABEL = _resolve_marker_panel(cfg.get("marker_panel"))
    print(f"[markers] panel={cfg.get('marker_panel', 'lung_adc')} → {len(MARKERS)} genes, legend='{MARKER_LABEL}'")

    # Mode resolution
    single_fold = bool(args.single_fold) or bool(cfg.get("single_fold", False))
    if not single_fold:
        donors = args.donors or cfg.get("donors") or ALL_DONORS
        bad = [d for d in donors if d not in ALL_DONORS]
        if bad:
            raise ValueError(f"Unknown donors: {bad}")
    else:
        donors = None

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData once (used by Part A for all folds). Path is config-driven so
    # lung uses gse131907 and CRC uses gse144735.
    anndata_path = Path(cfg.get("anndata_path", ANNDATA_PATH))
    print(f"\nLoading AnnData from {anndata_path} …")
    if not anndata_path.exists():
        raise FileNotFoundError(f"AnnData not found: {anndata_path}")
    adata = sc.read_h5ad(anndata_path)

    # Ensure BinaryClass + Patient columns match the labeling/ID rule the tokens were
    # built with, so cell filters here line up with splits_by_patient.npz.
    if "Cell_subtype" in adata.obs.columns:
        # Lung (GSE131907): apply the same rule as lung_tokenize.py.
        obs = adata.obs
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
        binary[is_normal & ~is_tumor] = "Normal"
        adata.obs["BinaryClass"] = binary
        def _patient_id(sample: str) -> str:
            m = re.search(r"(\d+)", str(sample))
            return m.group(1) if m else str(sample)
        adata.obs["Patient"] = obs["Sample"].astype(str).map(_patient_id)
    elif "BinaryClass" not in adata.obs.columns and "Class" in adata.obs.columns:
        # CRC (GSE144735): Class is already Tumor/Normal/Border.
        adata.obs["BinaryClass"] = adata.obs["Class"].replace({"Border": "Normal"})
    label_column = cfg["label_column"]
    print(f"AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Load token artifacts once (used by Part B for all donors)
    tokens_dir = Path(cfg["tokens_dir"])
    tokens_npz = np.load(find_single(tokens_dir, TOKEN_GLOB, "token npz"))
    tokens = tokens_npz["tokens"]
    meta_df = pd.read_csv(find_single(tokens_dir, META_GLOB, "metadata"), sep="\t")
    meta_df = ensure_label_column(meta_df, label_column)
    encoded_labels, id2label, label2id = prepare_label_mappings(meta_df[label_column])

    dataset_vocab = load_gene_vocab(tokens_dir / "gene_vocab.tsv")
    dataset_pad_fill_value = int(dataset_vocab.max()) + 1

    remap = None
    pad_fill_value = dataset_pad_fill_value
    model_pad_token_id = None
    model_token_to_gene: Dict[int, str] = {}

    if not args.skip_attention and cfg.get("model_vocab"):
        gene_name_dict = None
        if cfg.get("model_gene_name_dict"):
            gene_name_dict = load_gene_name_dict(Path(cfg["model_gene_name_dict"]))
        model_vocab = load_gene_vocab(Path(cfg["model_vocab"]))
        model_meta = AutoConfig.from_pretrained(str(cfg["model_name_or_path"]))
        model_pad_token_id = model_meta.pad_token_id
        if model_pad_token_id is None:
            raise ValueError("Model config has no pad_token_id.")
        remap = build_vocab_remap(
            dataset_vocab=dataset_vocab,
            model_vocab=model_vocab,
            pad_fill_value=dataset_pad_fill_value,
            pad_token_id=model_pad_token_id,
            unknown_token_id=model_pad_token_id,
            gene_name_map=gene_name_dict,
        )
        model_token_to_gene = build_model_token_to_gene(
            dataset_vocab, remap, model_pad_token_id
        )
        print(f"Gene→token map: {len(model_token_to_gene)} entries")

    device = _get_device()
    print(f"Device: {device}")

    # Build the (label, patient_ids, output_prefix, model_dir) list
    folds_to_run: list[tuple[str, list[str], str, Path]] = []
    if single_fold:
        splits_path = tokens_dir / "splits_by_patient.npz"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"{splits_path} not found. Run `python scripts/lung_split.py` first, "
                "or remove --single-fold to use LODO mode."
            )
        splits = np.load(splits_path, allow_pickle=True)
        key = f"{args.split_set}_patients"
        if key not in splits.files:
            raise KeyError(f"{splits_path} missing array '{key}'")
        patient_ids = [str(p) for p in splits[key].tolist()]
        output_prefix = (args.output_prefix
                         or cfg.get("output_prefix")
                         or f"lung_{args.split_set}")
        model_dir = Path(cfg["output_dir"])
        folds_to_run.append((output_prefix, patient_ids, output_prefix, model_dir))
        print(f"\n[single-fold] split={args.split_set}  patients={patient_ids}")
        print(f"[single-fold] output_prefix={output_prefix}")
    else:
        for donor in donors:
            folds_to_run.append((
                donor, [donor], f"fold_{donor}",
                Path(cfg["output_dir"]) / f"fold_{donor}",
            ))

    for label, patient_ids, output_prefix, model_dir in folds_to_run:
        sep = "-" * 50
        print(f"\n{sep}\n{label}\n{sep}")

        # ── Part A: expression ranking ──────────────────────────────────
        print("Part A — expression ranking …")
        expr_rankings = compute_expression_ranking(adata, patient_ids, label_column, label=label)
        plot_expression_ranking(expr_rankings, label, output_prefix, results_dir)

        # ── Part A.5: log2 fold-change (differential expression) ────────
        print("Part A.5 — log2 fold-change ranking …")
        logfc_rankings = compute_logfc_ranking(adata, patient_ids, label_column, label=label)
        plot_logfc_ranking(logfc_rankings, label, output_prefix, results_dir)
        save_logfc_tsv(logfc_rankings, output_prefix, results_dir)

        # ── Part B: attention ranking ───────────────────────────────────
        attn_rankings: Dict[str, pd.DataFrame] = {}
        if not args.skip_attention:
            print("Part B — attention ranking …")
            num_labels = len(id2label)
            model = _load_fold_model(
                model_dir,
                use_base_model=args.use_base_model,
                base_model_path=str(cfg["model_name_or_path"]),
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                device=device,
            )
            if model is not None:
                attn_rankings = extract_attention_rankings(
                    tokens=tokens,
                    metadata=meta_df,
                    encoded_labels=encoded_labels,
                    id2label=id2label,
                    label2id=label2id,
                    remap=remap,
                    pad_fill_value=pad_fill_value,
                    model=model,
                    model_token_to_gene=model_token_to_gene,
                    patient_ids=patient_ids,
                    label=label,
                    cfg=cfg,
                    device=device,
                )
                plot_attention_ranking(attn_rankings, label, output_prefix, results_dir)
                save_attention_tsv(attn_rankings, output_prefix, results_dir)
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            print("Part B — skipped (--skip-attention).")

        # ── Part C: combined figure ─────────────────────────────────────
        print("Part C — combined figure …")
        plot_combined(
            expr_rankings, attn_rankings, label, output_prefix, results_dir,
            logfc_rankings=logfc_rankings,
        )

    print(f"\nDone. Results in: {results_dir}")


if __name__ == "__main__":
    main()
