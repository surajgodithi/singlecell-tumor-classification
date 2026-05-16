#!/usr/bin/env python
"""
In Silico Perturbation Analysis for LODO folds.

Two-phase analysis per fold:

  Phase 1 — Sanity check with known CRC markers:
    Perturb KRAS, APC, TP53, MYC, EGFR, BRAF, EPCAM, CDX2, SMAD4, PIK3CA.

  Phase 2 — Discovery perturbation on top 200 attention genes:
    Perturb the top 200 genes from the attention ranking produced by
    gene_ranking_analysis.py. Falls back to token frequency if the TSV
    is absent.

For each gene on Tumor cells:
  - delta              = mean P(Tumor|perturbed) − mean P(Tumor|baseline) over ALL cells
                         (large negative ↓ → oncogene candidate)
  - delta_present      = same, but averaged ONLY over cells where the gene was actually
                         expressed (i.e. token present in the input sequence). This is the
                         biologically meaningful number for sparsely expressed genes —
                         `delta` dilutes a strong local effect with cells that never had
                         the gene to begin with.
  - flip_fraction      = fraction of Tumor cells whose prediction flipped Tumor → Normal
  - flip_fraction_present = same restricted to cells where the gene was present

For each gene on Normal cells: same shape, large positive `delta` → TSG candidate.

Output per fold: results/lodo/fold_{donor}_perturbation.tsv
Columns: gene, class, phase, n_cells, n_cells_with_gene,
         baseline_mean_prob, perturbed_mean_prob, delta, flip_fraction,
         baseline_mean_prob_present, perturbed_mean_prob_present,
         delta_present, flip_fraction_present

Usage
-----
# Fast smoke test — base model, 50 cells, 10 discovery genes only (~2 min):
    python scripts/in_silico_perturbation.py --donors KUL01 --use-base-model --max-cells 50 --top-n-discovery 10 --perturb-batch-size 32

# One real fold (requires outputs/lodo/fold_KUL01/ checkpoint from lodo_cv.py):
    python scripts/in_silico_perturbation.py --donors KUL01

# All folds:
    python scripts/in_silico_perturbation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
    softmax,
)
from gene_ranking_analysis import (
    ALL_DONORS,
    CRC_MARKERS,
    DEFAULT_CONFIG,
    FALLBACK_DEFAULTS,
    META_GLOB,
    TOKEN_GLOB,
    _get_device,
    _load_fold_model,
    build_model_token_to_gene,
    load_config,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known cancer driver gene panels. The config picks which list is "Phase 1" via
# `marker_panel: lung_adc | crc | both`.
LUNG_ADC_MARKERS: List[str] = [
    "EGFR", "KRAS", "ALK", "TP53", "STK11", "KEAP1",
    "NF1", "RBM10", "MET", "ERBB2",
]
CRC_MARKERS: List[str] = [
    "KRAS", "APC", "TP53", "MYC", "EGFR", "BRAF",
    "EPCAM", "CDX2", "SMAD4", "PIK3CA",
]
# Housekeeping genes — negative controls. Perturbing these should produce
# near-zero |delta|; the noise floor calibrates what "real" effects look like.
HOUSEKEEPERS: List[str] = [
    "ACTB", "GAPDH", "RPL13A", "B2M", "HPRT1",
    "PPIA", "TBP", "UBC", "YWHAZ", "SDHA",
]
# Back-compat alias (existing CRC-era code references KNOWN_MARKERS).
KNOWN_MARKERS: List[str] = CRC_MARKERS

DEFAULT_TOP_N_DISCOVERY = 200
DEFAULT_PERTURB_BATCH = 16
DEFAULT_SWEEP_MAX_K = 10          # cumulative knockout sweep depth
DEFAULT_PAIRWISE_TOP_N = 20       # pairs from top-N by |delta_present|

OUTPUT_COLS = [
    "fold_donor", "class", "gene", "phase",
    "n_cells", "n_cells_with_gene",
    # All-cells stats (preserved for back-compat / explicit dilution comparison)
    "baseline_mean_prob", "perturbed_mean_prob", "delta", "flip_fraction",
    # Present-only stats (biologically meaningful for sparsely expressed genes)
    "baseline_mean_prob_present", "perturbed_mean_prob_present",
    "delta_present", "flip_fraction_present",
]
# Extra columns added by the new phase types (sweep / pairwise). Not all rows
# carry these; aggregation downstream tolerates missing columns.
EXTRA_PHASE_COLS = [
    "k_or_pair",     # k value for sweep_k* rows; "A|B" pair string for pairwise rows
    "genes_in_set",  # comma-joined gene list for sweep/pairwise rows
]


def select_marker_panel(name: str) -> List[str]:
    """Resolve `marker_panel` config value to the actual gene list."""
    key = (name or "").lower()
    if key in ("lung", "lung_adc", "lung-adc", "adc"):
        return list(LUNG_ADC_MARKERS)
    if key == "crc":
        return list(CRC_MARKERS)
    if key == "both":
        # Preserve order, no dups
        seen, out = set(), []
        for g in LUNG_ADC_MARKERS + CRC_MARKERS:
            if g not in seen:
                seen.add(g); out.append(g)
        return out
    if key in ("", "none"):
        return []
    raise ValueError(f"Unknown marker_panel: {name!r} (expected lung_adc | crc | both | none)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--donors", nargs="+", metavar="DONOR", default=None,
                   help="LODO folds to run (CRC mode; default: all 6). Ignored if --single-fold.")
    p.add_argument("--single-fold", action="store_true",
                   help="Run on a single train/val/test split (lung mode). Uses test_idx from "
                        "splits_by_patient.npz; one TSV output named by `output_prefix` in config.")
    p.add_argument("--split-set", choices=["test", "val"], default="test",
                   help="In --single-fold mode, which split to perturb on (default: test).")
    p.add_argument("--output-prefix", type=str, default=None,
                   help="In --single-fold mode, output file prefix (default: config 'output_prefix' "
                        "or 'lung_test').")
    p.add_argument("--use-base-model", action="store_true",
                   help="Use base Geneformer weights instead of fold checkpoint (smoke test).")
    p.add_argument("--top-n-discovery", type=int, default=DEFAULT_TOP_N_DISCOVERY,
                   help=f"Genes from attention ranking for Phase 2 (default: {DEFAULT_TOP_N_DISCOVERY}).")
    p.add_argument("--sweep-max-k", type=int, default=DEFAULT_SWEEP_MAX_K,
                   help=f"Cumulative knockout-sweep depth K=1..N (default: {DEFAULT_SWEEP_MAX_K}). "
                        "Set to 0 to skip the sweep.")
    p.add_argument("--pairwise-top-n", type=int, default=DEFAULT_PAIRWISE_TOP_N,
                   help=f"Pairwise interaction matrix on top-N single-gene hits "
                        f"(default: {DEFAULT_PAIRWISE_TOP_N}; N*(N-1)/2 pairs). Set to 0 to skip.")
    p.add_argument("--marker-panel", choices=["lung_adc", "crc", "both", "none"], default=None,
                   help="Override known-marker panel for Phase 1 (default: from config).")
    p.add_argument("--include-housekeepers", dest="include_housekeepers",
                   action="store_true", default=None,
                   help="Include housekeeping genes as Phase 1 negative controls.")
    p.add_argument("--no-housekeepers", dest="include_housekeepers", action="store_false",
                   help="Disable housekeeper Phase 1 entries.")
    p.add_argument("--perturb-batch-size", type=int, default=DEFAULT_PERTURB_BATCH,
                   help=f"Batch size for perturbation inference (default: {DEFAULT_PERTURB_BATCH}).")
    p.add_argument("--max-cells", type=int, default=None,
                   help="Cap cells per class per fold — for fast smoke tests (e.g. --max-cells 50). "
                        "Omit for full production runs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gene set construction
# ---------------------------------------------------------------------------

def build_gene_sets(
    label: str,
    cell_indices: np.ndarray,
    results_dir: Path,
    output_prefix: str,
    top_n_discovery: int,
    marker_list: List[str],
    include_housekeepers: bool,
    model_token_to_gene: Dict[int, str],
    tokens: np.ndarray,
    remap: Optional[np.ndarray],
    pad_fill_value: int,
) -> List[Tuple[str, str]]:
    """
    Return list of (gene_symbol, phase) tuples:
      - Phase 1a: `marker_list` genes tagged "known"  (lung AdC or CRC drivers per config)
      - Phase 1b: HOUSEKEEPERS tagged "housekeeper"   (negative controls, optional)
      - Phase 2:  top_n_discovery attention genes tagged "discovery", de-duplicated
    `cell_indices` defines which cells to use for the token-frequency fallback when no
    attention TSV exists. `output_prefix` controls the per-fold/per-run filename.
    """
    # Phase 1a — known cancer drivers
    phase1 = [(g, "known") for g in marker_list]

    # Phase 1b — housekeeping negative controls
    if include_housekeepers:
        already = {g for g, _ in phase1}
        phase1_hk = [(g, "housekeeper") for g in HOUSEKEEPERS if g not in already]
    else:
        phase1_hk = []

    # Phase 2 — discovery genes from attention ranking
    discovery_genes = _load_discovery_genes(
        label, cell_indices, results_dir, output_prefix, top_n_discovery,
        model_token_to_gene, tokens, remap, pad_fill_value,
    )
    already = {g for g, _ in phase1 + phase1_hk}
    phase2 = [(g, "discovery") for g in discovery_genes if g not in already]

    all_genes = phase1 + phase1_hk + phase2
    print(f"  [{label}] Phase 1 known: {len(phase1)} | "
          f"Phase 1 housekeeper: {len(phase1_hk)} | "
          f"Phase 2 discovery: {len(phase2)} | Total: {len(all_genes)}")
    return all_genes


def _load_discovery_genes(
    label: str,
    cell_indices: np.ndarray,
    results_dir: Path,
    output_prefix: str,
    top_n: int,
    model_token_to_gene: Dict[int, str],
    tokens: np.ndarray,
    remap: Optional[np.ndarray],
    pad_fill_value: int,
) -> List[str]:
    """Top-N attention genes (preferred) or token frequency fallback."""
    attn_tsv = results_dir / f"{output_prefix}_attention_genes.tsv"
    if attn_tsv.exists():
        df = pd.read_csv(attn_tsv, sep="\t")
        if "mean_attn_tumor" in df.columns:
            genes = (
                df.sort_values("mean_attn_tumor", ascending=False, na_position="last")
                .head(top_n)["gene"]
                .tolist()
            )
            print(f"  [{label}] Discovery genes from attention TSV "
                  f"({len(genes)} genes, {attn_tsv.name})")
            return genes

    print(f"  [{label}] No attention TSV ({attn_tsv.name}) — token-frequency fallback.")
    max_len = tokens.shape[1]
    sub_seqs = tokens[cell_indices, :max_len].astype(np.int64)
    real_mask = sub_seqs != -1
    padded_seqs = sub_seqs.copy()
    padded_seqs[~real_mask] = pad_fill_value
    remapped_seqs = remap[padded_seqs] if remap is not None else padded_seqs
    real_toks = remapped_seqs[real_mask]
    tok_ids, counts = np.unique(real_toks, return_counts=True)
    gene_freq = [
        (model_token_to_gene[t], c)
        for t, c in zip(tok_ids.tolist(), counts.tolist())
        if t in model_token_to_gene
    ]
    gene_freq.sort(key=lambda x: x[1], reverse=True)
    return [g for g, _ in gene_freq[:top_n]]


# ---------------------------------------------------------------------------
# Baseline inference
# ---------------------------------------------------------------------------

def compute_baseline(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    tumor_label_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect model-space input_ids, attention masks, and P(Tumor) for every cell.

    Returns:
      all_input_ids  : (n_cells, seq_len) int64
      all_attn_masks : (n_cells, seq_len) int64
      baseline_probs : (n_cells,)          float32  P(Tumor) baseline
    """
    all_ids, all_masks, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=masks)
            probs = torch.softmax(out.logits, dim=-1)
            all_probs.append(probs[:, tumor_label_id].cpu().numpy())
            all_ids.append(batch["input_ids"].numpy())
            all_masks.append(batch["attention_mask"].numpy())
    return (
        np.concatenate(all_ids, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_probs).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Single-gene perturbation
# ---------------------------------------------------------------------------

def perturb_one_gene(
    all_input_ids: np.ndarray,
    all_attn_masks: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_preds: np.ndarray,
    gene_token_id: int,
    pad_token_id: int,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    tumor_label_id: int,
    batch_size: int,
    cls: str,
) -> dict:
    """
    Mask gene_token_id in every cell's sequence, re-run inference, compute stats.

    baseline_preds : (n_cells,) int — 0=Normal, 1=Tumor before perturbation
    cls            : "Tumor" or "Normal" (determines flip direction)
    """
    n_cells = all_input_ids.shape[0]
    perturbed_probs = np.empty(n_cells, dtype=np.float32)
    gene_present = np.zeros(n_cells, dtype=bool)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            ids_b = torch.from_numpy(all_input_ids[start:end].copy()).to(device)
            msk_b = torch.from_numpy(all_attn_masks[start:end].copy()).to(device)

            gene_pos = ids_b == gene_token_id
            gene_present[start:end] = gene_pos.any(dim=1).cpu().numpy()

            ids_p = ids_b.clone()
            msk_p = msk_b.clone()
            ids_p[gene_pos] = pad_token_id
            msk_p[gene_pos] = 0

            out = model(input_ids=ids_p, attention_mask=msk_p)
            probs = torch.softmax(out.logits, dim=-1)
            perturbed_probs[start:end] = probs[:, tumor_label_id].cpu().numpy()

    perturbed_preds = (perturbed_probs >= 0.5).astype(int)  # 0=Normal, 1=Tumor
    delta = perturbed_probs - baseline_probs
    n_with_gene = int(gene_present.sum())

    if cls == "Tumor":
        # flip: was Tumor (pred=1), became Normal (pred=0)
        flip_mask = (baseline_preds == tumor_label_id) & (perturbed_preds != tumor_label_id)
    else:
        # flip: was Normal (pred≠tumor), became Tumor (pred=tumor)
        flip_mask = (baseline_preds != tumor_label_id) & (perturbed_preds == tumor_label_id)

    flip_fraction = float(flip_mask.sum() / n_cells) if n_cells > 0 else float("nan")

    # Present-only stats. Removing a gene from cells where it wasn't expressed contributes 0
    # to delta and dilutes the population-level effect of sparsely-expressed but locally
    # important genes (e.g. stem-cell markers in ~10% of Tumor cells). The present-only
    # numbers are the biologically meaningful version of each statistic.
    if n_with_gene > 0:
        present_mask = gene_present
        baseline_mean_present = float(np.mean(baseline_probs[present_mask]))
        perturbed_mean_present = float(np.mean(perturbed_probs[present_mask]))
        delta_present = float(np.mean(delta[present_mask]))
        flip_present_count = int((flip_mask & present_mask).sum())
        flip_fraction_present = float(flip_present_count / n_with_gene)
    else:
        baseline_mean_present = float("nan")
        perturbed_mean_present = float("nan")
        delta_present = float("nan")
        flip_fraction_present = float("nan")

    return {
        "n_cells": n_cells,
        "n_cells_with_gene": n_with_gene,
        "baseline_mean_prob": float(np.mean(baseline_probs)),
        "perturbed_mean_prob": float(np.mean(perturbed_probs)),
        "delta": float(np.mean(delta)),
        "flip_fraction": flip_fraction,
        "baseline_mean_prob_present": baseline_mean_present,
        "perturbed_mean_prob_present": perturbed_mean_present,
        "delta_present": delta_present,
        "flip_fraction_present": flip_fraction_present,
    }


# ---------------------------------------------------------------------------
# Multi-gene perturbation — knockout sweep + pairwise interaction matrix
# ---------------------------------------------------------------------------

def perturb_gene_set(
    all_input_ids: np.ndarray,
    all_attn_masks: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_preds: np.ndarray,
    gene_token_ids: list[int],
    pad_token_id: int,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    tumor_label_id: int,
    batch_size: int,
    cls: str,
) -> dict:
    """Simultaneous knockout of *all* genes in `gene_token_ids` per cell.

    Biology motivation: single-gene perturbation misses pathway redundancy
    (RAS family, multiple WNT-pathway readouts the model has learned, etc.).
    Removing a set at once reveals whether top hits are independent contributions
    or redundant reads of the same underlying signal.

    "Present" here means at least ONE of the set's genes was in the input — i.e.
    the cell could have been affected by the joint knockout.
    """
    if not gene_token_ids:
        raise ValueError("gene_token_ids must be non-empty")

    n_cells = all_input_ids.shape[0]
    perturbed_probs = np.empty(n_cells, dtype=np.float32)
    set_present = np.zeros(n_cells, dtype=bool)
    gene_token_set = torch.tensor(list(set(gene_token_ids)), dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            ids_b = torch.from_numpy(all_input_ids[start:end].copy()).to(device)
            msk_b = torch.from_numpy(all_attn_masks[start:end].copy()).to(device)

            # gene_pos: (B, L) bool — true where input matches ANY token in set
            gene_pos = (ids_b.unsqueeze(-1) == gene_token_set).any(dim=-1)
            set_present[start:end] = gene_pos.any(dim=1).cpu().numpy()

            ids_p = ids_b.clone()
            msk_p = msk_b.clone()
            ids_p[gene_pos] = pad_token_id
            msk_p[gene_pos] = 0

            out = model(input_ids=ids_p, attention_mask=msk_p)
            probs = torch.softmax(out.logits, dim=-1)
            perturbed_probs[start:end] = probs[:, tumor_label_id].cpu().numpy()

    perturbed_preds = (perturbed_probs >= 0.5).astype(int)
    delta = perturbed_probs - baseline_probs
    n_with_set = int(set_present.sum())

    if cls == "Tumor":
        flip_mask = (baseline_preds == tumor_label_id) & (perturbed_preds != tumor_label_id)
    else:
        flip_mask = (baseline_preds != tumor_label_id) & (perturbed_preds == tumor_label_id)
    flip_fraction = float(flip_mask.sum() / n_cells) if n_cells > 0 else float("nan")

    if n_with_set > 0:
        present = set_present
        baseline_mean_present = float(np.mean(baseline_probs[present]))
        perturbed_mean_present = float(np.mean(perturbed_probs[present]))
        delta_present = float(np.mean(delta[present]))
        flip_fraction_present = float(int((flip_mask & present).sum()) / n_with_set)
    else:
        baseline_mean_present = float("nan")
        perturbed_mean_present = float("nan")
        delta_present = float("nan")
        flip_fraction_present = float("nan")

    return {
        "n_cells": n_cells, "n_cells_with_gene": n_with_set,
        "baseline_mean_prob": float(np.mean(baseline_probs)),
        "perturbed_mean_prob": float(np.mean(perturbed_probs)),
        "delta": float(np.mean(delta)), "flip_fraction": flip_fraction,
        "baseline_mean_prob_present": baseline_mean_present,
        "perturbed_mean_prob_present": perturbed_mean_present,
        "delta_present": delta_present, "flip_fraction_present": flip_fraction_present,
    }


def run_knockout_sweep(
    top_gene_symbols: list[str],
    gene_to_model_token: Dict[str, int],
    all_input_ids: np.ndarray,
    all_attn_masks: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_preds: np.ndarray,
    pad_token_id: int,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    tumor_label_id: int,
    batch_size: int,
    cls: str,
    max_k: int,
    label: str,
) -> list[dict]:
    """Cumulative knockout sweep — perturb top-1, top-2, ..., top-K simultaneously.

    Returns one row per K. Plotting `delta_present` vs K reveals redundancy:
    a plateau at K=3 means the model uses ~3 dominant signals; a linear decrease
    means the genes are independent contributions.
    """
    rows: list[dict] = []
    # Map symbols → model token ids, skipping any that aren't in vocab
    ordered_tokens: list[tuple[str, int]] = []
    for g in top_gene_symbols[:max_k]:
        tok = gene_to_model_token.get(g)
        if tok is not None:
            ordered_tokens.append((g, tok))
    if not ordered_tokens:
        return rows

    print(f"  [{label}/{cls}] knockout sweep K=1..{len(ordered_tokens)} …", flush=True)
    cumulative_tokens: list[int] = []
    cumulative_symbols: list[str] = []
    for k, (sym, tok) in enumerate(ordered_tokens, start=1):
        cumulative_tokens.append(tok)
        cumulative_symbols.append(sym)
        stats = perturb_gene_set(
            all_input_ids, all_attn_masks, baseline_probs, baseline_preds,
            cumulative_tokens, pad_token_id, model, device, tumor_label_id,
            batch_size, cls,
        )
        rows.append({
            "class": cls, "gene": f"sweep_top{k}",
            "phase": f"sweep_K{k}",
            "k_or_pair": k,
            "genes_in_set": ",".join(cumulative_symbols),
            **stats,
        })
        if k <= 3 or k == len(ordered_tokens):
            print(f"    [{label}/{cls}] K={k} ({sym}): "
                  f"delta_present={stats['delta_present']:+.4f}  "
                  f"flip_present={stats['flip_fraction_present']:.3f}", flush=True)
    return rows


def run_pairwise_matrix(
    top_gene_symbols: list[str],
    gene_to_model_token: Dict[str, int],
    all_input_ids: np.ndarray,
    all_attn_masks: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_preds: np.ndarray,
    pad_token_id: int,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    tumor_label_id: int,
    batch_size: int,
    cls: str,
    top_n: int,
    single_gene_deltas: Dict[str, float],
    label: str,
) -> list[dict]:
    """Pairwise interaction matrix on the top-N single-gene hits.

    For each pair (i, j) computes the epistasis term:
        ε = Δ_ij - (Δ_i + Δ_j)
    interpreted as:
        ε ≈ 0  → independent (different mechanisms — both real targets)
        ε < 0  → same pathway / redundant (one masks the other)
        ε > 0  → synergy — candidate combo target

    `single_gene_deltas` should be a {gene_symbol → delta_present} map from the
    Phase 2 single-gene results in the SAME class.
    """
    rows: list[dict] = []
    ordered = [
        (g, gene_to_model_token[g])
        for g in top_gene_symbols[:top_n]
        if g in gene_to_model_token
    ]
    if len(ordered) < 2:
        return rows

    n_pairs = len(ordered) * (len(ordered) - 1) // 2
    print(f"  [{label}/{cls}] pairwise matrix on top-{len(ordered)} "
          f"({n_pairs} pairs) …", flush=True)
    done = 0
    for i in range(len(ordered)):
        sym_i, tok_i = ordered[i]
        d_i = single_gene_deltas.get(sym_i, float("nan"))
        for j in range(i + 1, len(ordered)):
            sym_j, tok_j = ordered[j]
            d_j = single_gene_deltas.get(sym_j, float("nan"))
            stats = perturb_gene_set(
                all_input_ids, all_attn_masks, baseline_probs, baseline_preds,
                [tok_i, tok_j], pad_token_id, model, device, tumor_label_id,
                batch_size, cls,
            )
            d_ij = stats["delta_present"]
            epistasis = float(d_ij - (d_i + d_j)) if not (
                np.isnan(d_i) or np.isnan(d_j) or np.isnan(d_ij)
            ) else float("nan")
            rows.append({
                "class": cls, "gene": f"{sym_i}|{sym_j}",
                "phase": "pairwise",
                "k_or_pair": f"{sym_i}|{sym_j}",
                "genes_in_set": f"{sym_i},{sym_j}",
                "epistasis_eps": epistasis,
                "delta_i_alone": d_i, "delta_j_alone": d_j,
                **stats,
            })
            done += 1
            if done % 50 == 0:
                print(f"    [{label}/{cls}] pairs done: {done}/{n_pairs}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Per-fold routine
# ---------------------------------------------------------------------------

def run_perturbation_fold(
    label: str,
    patient_ids: list[str],
    output_prefix: str,
    model_dir: Path,
    tokens: np.ndarray,
    metadata: pd.DataFrame,
    encoded_labels: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    remap: Optional[np.ndarray],
    pad_fill_value: int,
    model_pad_token_id: int,
    model_token_to_gene: Dict[int, str],
    gene_to_model_token: Dict[str, int],
    cfg: dict,
    results_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run perturbation on a single fold or split.

    Args:
        label: short identifier for log messages (e.g. "KUL01" or "lung_test").
        patient_ids: cells where metadata.Patient ∈ patient_ids will be perturbed.
        output_prefix: filename prefix — e.g. "fold_KUL01" or "lung_test".
                       Results go to results_dir/{output_prefix}_perturbation.tsv.
        model_dir: directory holding best_checkpoint.txt (e.g. outputs/lodo/fold_KUL01
                   or outputs/lung).
    """
    sep = "=" * 60
    print(f"\n{sep}\nPerturbation — {label}\n  patients: {patient_ids}\n{sep}")

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
    if model is None:
        print(f"  [skip] No checkpoint for {label} — train first or pass --use-base-model.")
        return

    tumor_label_id = label2id["Tumor"]
    max_length = int(cfg["max_length"])
    perturb_batch = args.perturb_batch_size

    # Resolve Phase 1 marker panel + housekeepers from CLI or config
    panel_name = args.marker_panel or cfg.get("marker_panel", "lung_adc")
    marker_list = select_marker_panel(panel_name)
    include_hk = (args.include_housekeepers
                  if args.include_housekeepers is not None
                  else bool(cfg.get("include_housekeepers", True)))
    print(f"  [{label}] marker_panel={panel_name} ({len(marker_list)} markers), "
          f"include_housekeepers={include_hk}")

    # We need to compute the discovery genes from cells in the target set,
    # so build the cell-index list now (used for Phase 2 fallback ranking).
    in_split = metadata["Patient"].astype(str).isin([str(p) for p in patient_ids])
    target_idx_global = np.where(in_split)[0].astype(np.int64)

    gene_list = build_gene_sets(
        label, target_idx_global, results_dir, output_prefix,
        args.top_n_discovery, marker_list, include_hk,
        model_token_to_gene, tokens, remap, pad_fill_value,
    )

    all_rows: list[dict] = []

    for cls in ("Tumor", "Normal"):
        cls_mask = in_split & (metadata[cfg["label_column"]] == cls)
        cls_idx = np.where(cls_mask)[0].astype(np.int64)
        if len(cls_idx) == 0:
            print(f"  [warn] No {cls} cells for {label}.")
            continue

        if args.max_cells is not None and len(cls_idx) > args.max_cells:
            rng = np.random.default_rng(42)
            cls_idx = np.sort(rng.choice(cls_idx, size=args.max_cells, replace=False))
            print(f"  [{label}] {cls}: subsampled to {args.max_cells} cells (--max-cells)")

        print(f"\n  [{label}] {cls}: {len(cls_idx)} cells — computing baselines …")

        ds = RankedGeneDataset(
            tokens, cls_idx, encoded_labels, pad_fill_value,
            max_length, remap,
            token_mask_prob=0.0, mixup_prob=0.0, num_labels=num_labels,
        )
        collator = RankedGeneCollator(num_labels=num_labels)
        loader = DataLoader(
            ds, batch_size=perturb_batch, collate_fn=collator,
            shuffle=False, num_workers=0, pin_memory=False,
        )

        all_input_ids, all_attn_masks, baseline_probs = compute_baseline(
            model, loader, device, tumor_label_id
        )
        baseline_preds = (baseline_probs >= 0.5).astype(int)
        print(f"  [{label}] {cls} baseline — "
              f"mean P(Tumor)={baseline_probs.mean():.4f}, "
              f"std={baseline_probs.std():.4f}", flush=True)

        # ── Phase 1 + 2: single-gene perturbation ─────────────────────────
        cls_single_deltas: Dict[str, float] = {}
        for g_idx, (gene, phase) in enumerate(gene_list):
            tok_id = gene_to_model_token.get(gene)
            if tok_id is None:
                continue
            stats = perturb_one_gene(
                all_input_ids, all_attn_masks, baseline_probs, baseline_preds,
                tok_id, model_pad_token_id, model, device, tumor_label_id,
                perturb_batch, cls,
            )
            cls_single_deltas[gene] = stats.get("delta_present", float("nan"))
            if g_idx % 20 == 0 or phase in ("known", "housekeeper"):
                print(f"    [{label}/{cls}] {phase.upper():<12} {gene:<8} "
                      f"delta_present={stats.get('delta_present', float('nan')):+.4f} "
                      f"flip_present={stats.get('flip_fraction_present', float('nan')):.3f} "
                      f"({stats['n_cells_with_gene']}/{stats['n_cells']})", flush=True)
            all_rows.append({
                "fold_donor": label, "class": cls, "gene": gene, "phase": phase,
                **stats,
            })

        print(f"  [{label}] {cls}: Phase 1+2 ({len(gene_list)} genes) done.")

        # Top hits by |delta_present|, restricted to single-gene rows (excludes
        # housekeepers — those are negative controls, not candidate drivers)
        candidate_genes = [
            g for g, p in gene_list
            if p in ("known", "discovery") and g in cls_single_deltas
            and not np.isnan(cls_single_deltas[g])
        ]
        candidate_genes.sort(
            key=lambda g: abs(cls_single_deltas[g]), reverse=True
        )

        # ── Phase 3: knockout sweep ──────────────────────────────────────
        if args.sweep_max_k > 0 and candidate_genes:
            sweep_rows = run_knockout_sweep(
                candidate_genes, gene_to_model_token,
                all_input_ids, all_attn_masks, baseline_probs, baseline_preds,
                model_pad_token_id, model, device, tumor_label_id,
                perturb_batch, cls, int(args.sweep_max_k), label,
            )
            for r in sweep_rows:
                r["fold_donor"] = label
                all_rows.append(r)

        # ── Phase 4: pairwise interaction matrix ─────────────────────────
        if args.pairwise_top_n >= 2 and len(candidate_genes) >= 2:
            pair_rows = run_pairwise_matrix(
                candidate_genes, gene_to_model_token,
                all_input_ids, all_attn_masks, baseline_probs, baseline_preds,
                model_pad_token_id, model, device, tumor_label_id,
                perturb_batch, cls, int(args.pairwise_top_n),
                cls_single_deltas, label,
            )
            for r in pair_rows:
                r["fold_donor"] = label
                all_rows.append(r)

    if not all_rows:
        print(f"  [warn] No results for {label}.")
    else:
        df_out = pd.DataFrame(all_rows)
        keep_cols = [c for c in OUTPUT_COLS + EXTRA_PHASE_COLS
                     + ["epistasis_eps", "delta_i_alone", "delta_j_alone"]
                     if c in df_out.columns]
        df_out = df_out[keep_cols]
        tsv_path = results_dir / f"{output_prefix}_perturbation.tsv"
        df_out.to_csv(tsv_path, sep="\t", index=False)
        print(f"\n  Saved: {tsv_path}  ({len(df_out)} rows)")
        json_path = results_dir / f"{output_prefix}_perturbation.json"
        json_path.write_text(json.dumps(all_rows, indent=2, default=str))
        print(f"  Saved: {json_path}")

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    # Mode resolution: --single-fold or config single_fold=true switches off LODO
    # iteration and uses a single (train/val/test) split instead.
    single_fold = bool(args.single_fold) or bool(cfg.get("single_fold", False))
    if not single_fold:
        donors = args.donors or cfg.get("donors") or ALL_DONORS
        bad = [d for d in donors if d not in ALL_DONORS]
        if bad:
            raise ValueError(f"Unknown donors: {bad}. Valid: {ALL_DONORS}")
    else:
        donors = None  # not used in single_fold mode

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load shared token artifacts (once for all folds)
    tokens_dir = Path(cfg["tokens_dir"])
    tokens_npz = np.load(find_single(tokens_dir, TOKEN_GLOB, "token npz"))
    tokens = tokens_npz["tokens"]
    meta_df = pd.read_csv(find_single(tokens_dir, META_GLOB, "metadata"), sep="\t")
    label_column = cfg["label_column"]
    meta_df = ensure_label_column(meta_df, label_column)
    encoded_labels, id2label, label2id = prepare_label_mappings(meta_df[label_column])

    dataset_vocab = load_gene_vocab(tokens_dir / "gene_vocab.tsv")
    dataset_pad_fill_value = int(dataset_vocab.max()) + 1

    # Build vocab remap and gene↔token lookups
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
    gene_to_model_token: Dict[str, int] = {g: t for t, g in model_token_to_gene.items()}
    print(f"Gene↔token map: {len(model_token_to_gene)} entries")

    # Report which known markers are covered by the vocabulary
    missing_markers = [m for m in KNOWN_MARKERS if m not in gene_to_model_token]
    if missing_markers:
        print(f"[warn] Known markers not in model vocab: {missing_markers}")

    pad_fill_value = dataset_pad_fill_value
    device = _get_device()
    print(f"Device: {device}")
    print(f"Phase 2 top-N: {args.top_n_discovery}")
    print(f"Phase 3 sweep K: 1..{args.sweep_max_k}" if args.sweep_max_k > 0 else "Phase 3: disabled")
    print(f"Phase 4 pairwise: top-{args.pairwise_top_n}" if args.pairwise_top_n >= 2 else "Phase 4: disabled")
    if args.max_cells:
        print(f"max-cells cap: {args.max_cells} (smoke test mode)")

    # Build the (label, patient_ids, output_prefix, model_dir) list for this run
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
            raise KeyError(f"{splits_path} missing array '{key}' (expected: train_patients / val_patients / test_patients)")
        patient_ids = [str(p) for p in splits[key].tolist()]
        output_prefix = (args.output_prefix
                         or cfg.get("output_prefix")
                         or f"lung_{args.split_set}")
        model_dir = Path(cfg["output_dir"])
        folds_to_run.append((output_prefix, patient_ids, output_prefix, model_dir))
        print(f"\n[single-fold] split={args.split_set}  patients={patient_ids}")
        print(f"[single-fold] output_prefix={output_prefix}  model_dir={model_dir}")
    else:
        print(f"Donors: {donors}")
        for donor in donors:
            folds_to_run.append((
                donor, [donor], f"fold_{donor}",
                Path(cfg["output_dir"]) / f"fold_{donor}",
            ))

    for label, patient_ids, output_prefix, model_dir in folds_to_run:
        run_perturbation_fold(
            label=label,
            patient_ids=patient_ids,
            output_prefix=output_prefix,
            model_dir=model_dir,
            tokens=tokens,
            metadata=meta_df,
            encoded_labels=encoded_labels,
            id2label=id2label,
            label2id=label2id,
            remap=remap,
            pad_fill_value=pad_fill_value,
            model_pad_token_id=model_pad_token_id,
            model_token_to_gene=model_token_to_gene,
            gene_to_model_token=gene_to_model_token,
            cfg=cfg,
            results_dir=results_dir,
            args=args,
            device=device,
        )

    print(f"\nAll perturbations complete. Results in: {results_dir}")


if __name__ == "__main__":
    main()
