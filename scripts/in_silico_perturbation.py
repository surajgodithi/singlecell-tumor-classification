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
  - mean_delta  = mean P(Tumor|perturbed) − mean P(Tumor|baseline)
                  (large negative ↓ → oncogene candidate)
  - flip_fraction = fraction of Tumor cells whose prediction flipped
                    Tumor → Normal after removal

For each gene on Normal cells:
  - mean_delta  = mean P(Tumor|perturbed) − mean P(Tumor|baseline)
                  (large positive ↑ → TSG candidate)
  - flip_fraction = fraction of Normal cells that flipped Normal → Tumor

Output per fold: results/lodo/fold_{donor}_perturbation.tsv
Columns: gene, class, phase, baseline_mean_prob, perturbed_mean_prob,
         delta, flip_fraction, n_cells, n_cells_with_gene

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
KNOWN_MARKERS: List[str] = [
    "KRAS", "APC", "TP53", "MYC", "EGFR", "BRAF",
    "EPCAM", "CDX2", "SMAD4", "PIK3CA",
]
DEFAULT_TOP_N_DISCOVERY = 200
DEFAULT_PERTURB_BATCH = 16

OUTPUT_COLS = [
    "fold_donor", "class", "gene", "phase",
    "baseline_mean_prob", "perturbed_mean_prob",
    "delta", "flip_fraction",
    "n_cells", "n_cells_with_gene",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--donors", nargs="+", metavar="DONOR", default=None,
                   help="Folds to run (default: all 6).")
    p.add_argument("--use-base-model", action="store_true",
                   help="Use base Geneformer weights instead of fold checkpoint (smoke test).")
    p.add_argument("--top-n-discovery", type=int, default=DEFAULT_TOP_N_DISCOVERY,
                   help=f"Genes from attention ranking for Phase 2 (default: {DEFAULT_TOP_N_DISCOVERY}).")
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
    donor: str,
    results_dir: Path,
    top_n_discovery: int,
    model_token_to_gene: Dict[int, str],
    tokens: np.ndarray,
    metadata: pd.DataFrame,
    label_column: str,
    remap: Optional[np.ndarray],
    pad_fill_value: int,
) -> List[Tuple[str, str]]:
    """
    Return list of (gene_symbol, phase) tuples:
      - Phase 1: KNOWN_MARKERS regardless of attention ranking
      - Phase 2: top top_n_discovery attention genes not already in Phase 1
    """
    # Phase 1 — known CRC markers
    phase1 = [(g, "known") for g in KNOWN_MARKERS]

    # Phase 2 — discovery genes from attention ranking
    discovery_genes = _load_discovery_genes(
        donor, results_dir, top_n_discovery,
        model_token_to_gene, tokens, metadata, label_column, remap, pad_fill_value,
    )
    known_set = set(KNOWN_MARKERS)
    phase2 = [(g, "discovery") for g in discovery_genes if g not in known_set]

    all_genes = phase1 + phase2
    print(f"  [{donor}] Phase 1: {len(phase1)} known markers | "
          f"Phase 2: {len(phase2)} discovery genes | Total: {len(all_genes)}")
    return all_genes


def _load_discovery_genes(
    donor: str,
    results_dir: Path,
    top_n: int,
    model_token_to_gene: Dict[int, str],
    tokens: np.ndarray,
    metadata: pd.DataFrame,
    label_column: str,
    remap: Optional[np.ndarray],
    pad_fill_value: int,
) -> List[str]:
    """Top-N attention genes (preferred) or token frequency fallback."""
    attn_tsv = results_dir / f"fold_{donor}_attention_genes.tsv"
    if attn_tsv.exists():
        df = pd.read_csv(attn_tsv, sep="\t")
        if "mean_attn_tumor" in df.columns:
            genes = (
                df.sort_values("mean_attn_tumor", ascending=False, na_position="last")
                .head(top_n)["gene"]
                .tolist()
            )
            print(f"  [{donor}] Discovery genes loaded from attention TSV "
                  f"({len(genes)} genes, {attn_tsv.name})")
            return genes

    print(f"  [{donor}] No attention TSV — using token-frequency fallback for discovery genes.")
    donor_idx = np.where(metadata["Patient"] == donor)[0]
    max_len = tokens.shape[1]
    # Vectorised frequency count over all donor cells
    donor_seqs = tokens[donor_idx, :max_len].astype(np.int64)
    real_mask = donor_seqs != -1
    padded_seqs = donor_seqs.copy()
    padded_seqs[~real_mask] = pad_fill_value
    if remap is not None:
        remapped_seqs = remap[padded_seqs]
    else:
        remapped_seqs = padded_seqs
    real_toks = remapped_seqs[real_mask]
    tok_ids, counts = np.unique(real_toks, return_counts=True)
    token_counts = dict(zip(tok_ids.tolist(), counts.tolist()))
    gene_freq = [
        (model_token_to_gene[t], c)
        for t, c in token_counts.items()
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

    return {
        "n_cells": n_cells,
        "n_cells_with_gene": n_with_gene,
        "baseline_mean_prob": float(np.mean(baseline_probs)),
        "perturbed_mean_prob": float(np.mean(perturbed_probs)),
        "delta": float(np.mean(delta)),
        "flip_fraction": flip_fraction,
    }


# ---------------------------------------------------------------------------
# Per-fold routine
# ---------------------------------------------------------------------------

def run_perturbation_fold(
    donor: str,
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
    sep = "=" * 60
    print(f"\n{sep}\nPerturbation — held-out donor: {donor}\n{sep}")

    num_labels = len(id2label)
    fold_out = Path(cfg["output_dir"]) / f"fold_{donor}"

    model = _load_fold_model(
        fold_out,
        use_base_model=args.use_base_model,
        base_model_path=str(cfg["model_name_or_path"]),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        device=device,
    )
    if model is None:
        print(f"  [skip] No checkpoint for {donor} — run lodo_cv.py first or pass --use-base-model.")
        return

    tumor_label_id = label2id["Tumor"]
    max_length = int(cfg["max_length"])
    perturb_batch = args.perturb_batch_size

    gene_list = build_gene_sets(
        donor, results_dir, args.top_n_discovery,
        model_token_to_gene, tokens, metadata, cfg["label_column"],
        remap, pad_fill_value,
    )

    all_rows: list[dict] = []

    for cls in ("Tumor", "Normal"):
        cls_mask = (metadata["Patient"] == donor) & (metadata[cfg["label_column"]] == cls)
        cls_idx = np.where(cls_mask)[0].astype(np.int64)
        if len(cls_idx) == 0:
            print(f"  [warn] No {cls} cells for donor {donor}.")
            continue

        if args.max_cells is not None and len(cls_idx) > args.max_cells:
            rng = np.random.default_rng(42)
            cls_idx = np.sort(rng.choice(cls_idx, size=args.max_cells, replace=False))
            print(f"  [{donor}] {cls}: subsampled to {args.max_cells} cells (--max-cells)")

        print(f"\n  [{donor}] {cls}: {len(cls_idx)} cells — computing baselines …")

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
        print(f"  [{donor}] {cls} baseline — "
              f"mean P(Tumor)={baseline_probs.mean():.4f}, "
              f"std={baseline_probs.std():.4f}", flush=True)

        for g_idx, (gene, phase) in enumerate(gene_list):
            tok_id = gene_to_model_token.get(gene)
            if tok_id is None:
                continue

            stats = perturb_one_gene(
                all_input_ids, all_attn_masks,
                baseline_probs, baseline_preds,
                tok_id, model_pad_token_id,
                model, device, tumor_label_id,
                perturb_batch, cls,
            )

            if g_idx % 20 == 0 or phase == "known":
                print(f"    [{donor}/{cls}] {phase.upper()} {gene}: "
                      f"delta={stats['delta']:+.4f}, "
                      f"flip={stats['flip_fraction']:.3f}, "
                      f"present={stats['n_cells_with_gene']}/{stats['n_cells']}",
                      flush=True)

            all_rows.append({
                "fold_donor": donor,
                "class": cls,
                "gene": gene,
                "phase": phase,
                **stats,
            })

        print(f"  [{donor}] {cls}: all genes perturbed.")

    if not all_rows:
        print(f"  [warn] No results for {donor}.")
    else:
        df_out = pd.DataFrame(all_rows)
        df_out = df_out[[c for c in OUTPUT_COLS if c in df_out.columns]]
        tsv_path = results_dir / f"fold_{donor}_perturbation.tsv"
        df_out.to_csv(tsv_path, sep="\t", index=False)
        print(f"\n  Saved: {tsv_path}")

        json_path = results_dir / f"fold_{donor}_perturbation.json"
        json_path.write_text(json.dumps(all_rows, indent=2))
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

    donors = args.donors or cfg.get("donors") or ALL_DONORS
    bad = [d for d in donors if d not in ALL_DONORS]
    if bad:
        raise ValueError(f"Unknown donors: {bad}. Valid: {ALL_DONORS}")

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
    print(f"Donors: {donors}")
    print(f"Phase 2 top-N: {args.top_n_discovery}")
    if args.max_cells:
        print(f"max-cells cap: {args.max_cells} (smoke test mode)")

    for donor in donors:
        run_perturbation_fold(
            donor=donor,
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
