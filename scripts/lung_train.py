#!/usr/bin/env python
"""
Lung Geneformer fine-tune — single train/val/test split (no LODO).

Reuses the speed/memory stack built for LODO (length-bucketed sampler, dynamic
padding, gradient checkpointing, SDPA, autocast hooks) but trains exactly one
model on the patient-level 70/15/15 split produced by `scripts/lung_split.py`.

Pipeline:
    1. Load tokens + metadata + splits from `gse131907/processed/tokens/`.
    2. Build RankedGeneDatasets for train/val/test.
    3. Class-donor weighted sampling on the train indices (handles class
       imbalance even on the cleanest 24784:3703 Tumor:Normal pool).
    4. Train with early stopping on val macro_f1.
    5. Evaluate best checkpoint on val and test; write per-class metrics for both.

Outputs:
    outputs/lung/best_checkpoint.txt         — pointer to the best HF checkpoint dir
    results/lung/lung_val_metrics.json       — full per-class metrics on val
    results/lung/lung_test_metrics.json      — full per-class metrics on test (held-out)
    results/lung/lung_summary.tsv            — one-row summary of headline numbers

Run:
    python scripts/lung_train.py --dry-run        # validate data pipeline, ~30 s
    python scripts/lung_train.py --num-epochs 1   # quick smoke test
    python scripts/lung_train.py                  # full run
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score,
)
from transformers import (
    AutoConfig, AutoModelForSequenceClassification,
    EarlyStoppingCallback, TrainingArguments, set_seed,
)

sys.path.insert(0, str(Path(__file__).parent))
from finetune_transformer import (
    RankedGeneDataset, build_sample_weights, build_vocab_remap,
    ensure_label_column, find_single, load_gene_name_dict,
    load_gene_vocab, prepare_label_mappings, resolve_class_weights, softmax,
)
from lodo_cv import (
    DynamicPadCollator, FastLodoTrainer,
    _compute_cell_lengths, _resolve_amp_dtype,
)

DEFAULT_CONFIG = Path("configs/lung_train.yaml")
TOKEN_GLOB = "*_gene_rank_tokens.npz"
META_GLOB = "*_tokens_metadata.tsv"

FALLBACK_DEFAULTS: dict = {
    "tokens_dir": "gse131907/processed/tokens",
    "output_dir": "outputs/lung",
    "results_dir": "results/lung",
    "model_name_or_path": "Geneformer/Geneformer-V2-104M",
    "model_vocab": "Geneformer/geneformer/token_dictionary_gc104M.pkl",
    "model_gene_name_dict": "Geneformer/geneformer/gene_name_id_dict_gc104M.pkl",
    "label_column": "BinaryClass",
    "focal_gamma": 1.0,
    "class_weights": {"Normal": 1.0, "Tumor": 2.05},
    "warmup_ratio": 0.10,
    "token_mask_prob": 0.01,
    "mixup_prob": 0.0,
    "mixup_alpha": 0.4,
    "balance_strategy": "class_donor",  # Patient column treated as "donor"
    "patience": 2,
    "num_epochs": 5,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "eval_accumulation_steps": 16,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2.0e-5,
    "weight_decay": 0.01,
    "seed": 42,
    "max_length": 2048,
    "amp_dtype": "none",  # bf16+SDPA+ckpt unstable on MPS 2.11 (see runs.md)
    "attn_implementation": "sdpa",
    "length_bucketing": True,
    "bucket_mega_factor": 50,
    "pad_to_multiple_of": 8,
    "save_total_limit": 2,
    "gradient_checkpointing": True,
}


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Check data pipeline shape and exit (no training).")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cfg: dict = {}
    if path and path.exists():
        cfg = yaml.safe_load(path.read_text()) or {}
        print(f"Loaded config: {path}")
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    for k, v in FALLBACK_DEFAULTS.items():
        cfg.setdefault(k, v)
    for field in ("tokens_dir", "output_dir", "results_dir", "model_vocab", "model_gene_name_dict"):
        if cfg.get(field) is not None:
            cfg[field] = Path(cfg[field])
    return cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _f1_only(eval_pred) -> dict:
    """Minimal compute_metrics for early stopping."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_detailed_metrics(
    logits: np.ndarray, labels: np.ndarray, id2label: Dict[int, str]
) -> dict:
    preds = logits.argmax(axis=-1)
    label_names = [id2label[i] for i in sorted(id2label)]
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    try:
        probs = softmax(logits)
        if len(id2label) == 2:
            pos_idx = next(i for i, n in id2label.items() if n == "Tumor")
            macro_auc = float(roc_auc_score(labels, probs[:, pos_idx]))
        else:
            macro_auc = float(roc_auc_score(labels, probs, multi_class="ovo", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    report = classification_report(
        labels, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall":    float(report[name]["recall"]),
            "f1":        float(report[name]["f1-score"]),
            "support":   int(report[name]["support"]),
        }
        for name in label_names if name in report
    }
    cm = confusion_matrix(
        labels, preds, labels=list(range(len(label_names)))
    ).tolist()
    return {
        "accuracy": acc, "macro_f1": macro_f1, "macro_roc_auc": macro_auc,
        "per_class": per_class, "confusion_matrix": cm, "label_names": label_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])

    # Load shared artifacts
    tokens_dir = Path(cfg["tokens_dir"])
    tokens_npz = np.load(find_single(tokens_dir, TOKEN_GLOB, "token npz"))
    tokens = tokens_npz["tokens"]
    if tokens.shape[1] < int(cfg["max_length"]):
        raise ValueError(
            f"max_length={cfg['max_length']} exceeds token width {tokens.shape[1]}"
        )

    if "lengths" in tokens_npz.files:
        cell_lengths = np.minimum(
            tokens_npz["lengths"].astype(np.int64), int(cfg["max_length"])
        )
    else:
        cell_lengths = np.minimum(_compute_cell_lengths(tokens), int(cfg["max_length"]))

    metadata = pd.read_csv(find_single(tokens_dir, META_GLOB, "metadata"), sep="\t")
    metadata = ensure_label_column(metadata, cfg["label_column"])
    encoded_labels, id2label, label2id = prepare_label_mappings(metadata[cfg["label_column"]])

    splits_path = tokens_dir / "splits_by_patient.npz"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found. Run `python scripts/lung_split.py` first."
        )
    splits = np.load(splits_path, allow_pickle=True)
    train_idx = splits["train_idx"].astype(np.int64)
    val_idx   = splits["val_idx"].astype(np.int64)
    test_idx  = splits["test_idx"].astype(np.int64)

    print(f"\nTokens : {tokens.shape}")
    print(f"Labels : {label2id}")
    print(f"Split  : train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    # Vocab remap
    dataset_vocab = load_gene_vocab(tokens_dir / "gene_vocab.tsv")
    dataset_pad_fill_value = int(dataset_vocab.max()) + 1
    gene_name_dict = None
    if cfg.get("model_gene_name_dict"):
        gene_name_dict = load_gene_name_dict(Path(cfg["model_gene_name_dict"]))
    remap = None
    pad_fill_value = dataset_pad_fill_value
    if cfg.get("model_vocab"):
        model_vocab = load_gene_vocab(Path(cfg["model_vocab"]))
        model_meta = AutoConfig.from_pretrained(str(cfg["model_name_or_path"]))
        model_pad = model_meta.pad_token_id
        if model_pad is None:
            raise ValueError("Model config has no pad_token_id.")
        remap = build_vocab_remap(
            dataset_vocab=dataset_vocab, model_vocab=model_vocab,
            pad_fill_value=dataset_pad_fill_value, pad_token_id=model_pad,
            unknown_token_id=model_pad, gene_name_map=gene_name_dict,
        )

    num_labels = len(id2label)
    max_length = int(cfg["max_length"])

    def make_ds(idx: np.ndarray, training: bool) -> RankedGeneDataset:
        return RankedGeneDataset(
            tokens, idx, encoded_labels, pad_fill_value, max_length, remap,
            token_mask_prob=float(cfg["token_mask_prob"]) if training else 0.0,
            mixup_prob=float(cfg["mixup_prob"]) if training else 0.0,
            mixup_alpha=float(cfg["mixup_alpha"]),
            num_labels=num_labels, rng_seed=cfg["seed"] if training else None,
        )

    train_ds = make_ds(train_idx, training=True)
    val_ds   = make_ds(val_idx,   training=False)
    test_ds  = make_ds(test_idx,  training=False)

    collator = DynamicPadCollator(
        num_labels=num_labels, max_length=max_length,
        pad_to_multiple_of=int(cfg["pad_to_multiple_of"]),
    )

    train_lengths = cell_lengths[train_idx]
    val_lengths   = cell_lengths[val_idx]
    test_lengths  = cell_lengths[test_idx]

    if args.dry_run:
        batch = collator([train_ds[0], train_ds[1]])
        print(f"\n[dry-run] input_ids={tuple(batch['input_ids'].shape)} "
              f"labels={batch['labels'].tolist()}")
        print("[dry-run] Pipeline OK, exiting.")
        return

    # Sample weights for class_donor balance (Patient column as the donor)
    sampler_weights = build_sample_weights(
        train_idx, metadata, cfg["label_column"], cfg["balance_strategy"]
    )
    if sampler_weights is not None:
        print(f"[sampler] class_donor weights: mean={sampler_weights.mean():.3f} "
              f"std={sampler_weights.std():.3f}")

    # Model
    model_config = AutoConfig.from_pretrained(
        str(cfg["model_name_or_path"]),
        num_labels=num_labels, id2label=id2label, label2id=label2id,
    )
    attn_impl = str(cfg.get("attn_implementation", "sdpa") or "eager")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(cfg["model_name_or_path"]), config=model_config,
            attn_implementation=attn_impl,
        )
        print(f"[model] loaded with attn_implementation={attn_impl}")
    except (TypeError, ValueError) as exc:
        print(f"[model] attn_implementation={attn_impl} unsupported ({exc}); using default.")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(cfg["model_name_or_path"]), config=model_config,
        )

    output_dir = Path(cfg["output_dir"])
    results_dir = Path(cfg["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    class_weights = resolve_class_weights(cfg.get("class_weights"), label2id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        eval_accumulation_steps=int(cfg["eval_accumulation_steps"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=int(cfg["seed"]),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    amp_dtype = _resolve_amp_dtype(cfg.get("amp_dtype"))
    if amp_dtype is not None:
        print(f"[amp] autocast enabled — {str(amp_dtype).split('.')[-1]}")

    use_bucketing = bool(cfg.get("length_bucketing", True)) and sampler_weights is not None
    if use_bucketing:
        print(f"[sampler] length-bucketed (mega_factor={cfg['bucket_mega_factor']})")

    trainer = FastLodoTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=_f1_only,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["patience"]))],
        class_weights=class_weights,
        sampler_weights=sampler_weights,
        focal_gamma=float(cfg["focal_gamma"]),
        train_lengths=train_lengths if use_bucketing else None,
        eval_lengths=val_lengths,
        bucket_mega_factor=int(cfg["bucket_mega_factor"]),
        amp_dtype=amp_dtype,
    )

    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint or str(output_dir)
    print(f"\nBest checkpoint: {best_ckpt}")
    (output_dir / "best_checkpoint.txt").write_text(best_ckpt)

    # --- Final evaluation on val and test (both held-out from training gradient) ---
    print("\n=== Final eval on val ===")
    trainer.eval_lengths = val_lengths
    val_pred = trainer.predict(val_ds)
    val_metrics = compute_detailed_metrics(
        val_pred.predictions, val_pred.label_ids, id2label
    )
    val_metrics["n_cells"] = len(val_idx)

    print("\n=== Final eval on test (held-out) ===")
    trainer.eval_lengths = test_lengths
    test_pred = trainer.predict(test_ds)
    test_metrics = compute_detailed_metrics(
        test_pred.predictions, test_pred.label_ids, id2label
    )
    test_metrics["n_cells"] = len(test_idx)

    (results_dir / "lung_val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    (results_dir / "lung_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\nval  → {results_dir / 'lung_val_metrics.json'}")
    print(f"test → {results_dir / 'lung_test_metrics.json'}")

    # Summary row
    def row(split: str, m: dict) -> dict:
        pc = m["per_class"]
        t = pc.get("Tumor", {}); n = pc.get("Normal", {})
        return {
            "split": split, "n_cells": m["n_cells"],
            "accuracy":      round(m["accuracy"], 4),
            "macro_f1":      round(m["macro_f1"], 4),
            "macro_roc_auc": round(m["macro_roc_auc"], 4),
            "tumor_precision": round(t.get("precision", 0), 4),
            "tumor_recall":    round(t.get("recall", 0), 4),
            "tumor_f1":        round(t.get("f1", 0), 4),
            "normal_precision": round(n.get("precision", 0), 4),
            "normal_recall":    round(n.get("recall", 0), 4),
            "normal_f1":        round(n.get("f1", 0), 4),
        }
    summary = pd.DataFrame([row("val", val_metrics), row("test", test_metrics)])
    summary_path = results_dir / "lung_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary → {summary_path}")
    print(summary.to_string(index=False))

    # Target check
    test_tumor_r = test_metrics["per_class"].get("Tumor", {}).get("recall", 0.0)
    test_normal_r = test_metrics["per_class"].get("Normal", {}).get("recall", 0.0)
    test_mf1 = test_metrics["macro_f1"]
    passed = test_mf1 > 0.65 and test_tumor_r > 0.70 and test_normal_r > 0.65
    print(f"\nTargets on test (macro_f1>0.65 AND tumor_recall>0.70 AND normal_recall>0.65): "
          f"{'PASS' if passed else 'MISS'}")
    print(f"  macro_f1={test_mf1:.4f}  tumor_recall={test_tumor_r:.4f}  normal_recall={test_normal_r:.4f}")


if __name__ == "__main__":
    main()
