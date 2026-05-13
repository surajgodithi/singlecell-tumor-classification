#!/usr/bin/env python
"""
Leave-One-Donor-Out (LODO) cross-validation for Geneformer CRC binary classification.

For each of the 6 donors, trains on the remaining 5 and evaluates on the held-out
donor. Each fold always initializes from base Geneformer weights — never from a
prior fine-tuned checkpoint.

Usage
-----
# 1. Validate the data pipeline without training (~30 s):
    python scripts/lodo_cv.py --dry-run

# 2. Quick smoke-test: one fold, one epoch (~5-10 min on MPS):
    python scripts/lodo_cv.py --donors KUL01 --num-epochs 1

# 3. Full run — all 6 folds:
    python scripts/lodo_cv.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).parent))
from finetune_transformer import (
    RankedGeneCollator,
    RankedGeneDataset,
    RankedGeneTrainer,
    build_sample_weights,
    build_vocab_remap,
    ensure_label_column,
    find_single,
    load_gene_name_dict,
    load_gene_vocab,
    prepare_label_mappings,
    resolve_class_weights,
    softmax,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Path("configs/lodo_cv.yaml")
TOKEN_GLOB = "*_gene_rank_tokens.npz"
META_GLOB = "*_tokens_metadata.tsv"
ALL_DONORS = ["KUL01", "KUL19", "KUL21", "KUL28", "KUL30", "KUL31"]
RUNS_LOG = Path("docs/runs.md")
SUMMARY_COLS = [
    "fold_donor", "n_train_cells", "n_val_cells",
    "accuracy", "macro_f1", "macro_roc_auc",
    "tumor_precision", "tumor_recall", "tumor_f1",
    "normal_precision", "normal_recall", "normal_f1",
    "best_checkpoint",
]
FALLBACK_DEFAULTS: dict = {
    "tokens_dir": "gse144735/processed/tokens",
    "output_dir": "outputs/lodo",
    "results_dir": "results/lodo",
    "model_name_or_path": "Geneformer/Geneformer-V2-104M",
    "model_vocab": "Geneformer/geneformer/token_dictionary_gc104M.pkl",
    "model_gene_name_dict": "Geneformer/geneformer/gene_name_id_dict_gc104M.pkl",
    "label_column": "BinaryClass",
    "focal_gamma": 1.0,
    "class_weights": {"Normal": 1.0, "Tumor": 2.05},
    "warmup_ratio": 0.10,
    "token_mask_prob": 0.015,
    "mixup_prob": 0.0,
    "mixup_alpha": 0.4,
    "balance_strategy": "class_donor",
    "patience": 3,
    "num_epochs": 8,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2.0e-5,
    "weight_decay": 0.01,
    "seed": 42,
    "max_length": 2048,
    "donors": ALL_DONORS,
}


# ---------------------------------------------------------------------------
# CLI / config resolution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config path (default: configs/lodo_cv.yaml).")
    p.add_argument("--donors", nargs="+", metavar="DONOR", default=None,
                   help="Which donors to use as held-out folds. Default: all 6.")
    p.add_argument("--dry-run", action="store_true",
                   help="Check data pipeline and batch shapes only — skip training.")
    p.add_argument("--num-epochs", type=int, default=None,
                   help="Override num_epochs from config (useful for smoke tests).")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def resolve_config(args: argparse.Namespace) -> dict:
    config_path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cfg = _load_yaml(config_path)
    if config_path and config_path.exists():
        print(f"Loaded config: {config_path}")

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
# Fold utilities
# ---------------------------------------------------------------------------

def build_fold_indices(
    metadata: pd.DataFrame, held_out_donor: str
) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = metadata["Patient"] != held_out_donor
    val_mask = metadata["Patient"] == held_out_donor
    return (
        np.where(train_mask)[0].astype(np.int64),
        np.where(val_mask)[0].astype(np.int64),
    )


def _f1_only(eval_pred) -> dict:
    """Minimal compute_metrics for HF Trainer — only used for early stopping."""
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
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in label_names
        if name in report
    }

    cm = confusion_matrix(
        labels, preds, labels=list(range(len(label_names)))
    ).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_auc,
        "per_class": per_class,
        "confusion_matrix": cm,
        "label_names": label_names,
    }


# ---------------------------------------------------------------------------
# Logging / persistence
# ---------------------------------------------------------------------------

def append_runs_log(
    donor: str, metrics: dict, cfg: dict, n_train: int, n_val: int
) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pc = metrics.get("per_class", {})
    t = pc.get("Tumor", {})
    n = pc.get("Normal", {})
    cw = cfg.get("class_weights") or {}

    entry = (
        f"\n### LODO fold: {donor} ({ts})\n"
        f"- **Train:** {', '.join(d for d in ALL_DONORS if d != donor)}"
        f" — {n_train:,} cells\n"
        f"- **Val:** {donor} — {n_val:,} cells\n"
        f"- focal_gamma={cfg.get('focal_gamma')} | "
        f"Tumor weight={cw.get('Tumor')} | "
        f"warmup_ratio={cfg.get('warmup_ratio')} | "
        f"token_mask_prob={cfg.get('token_mask_prob')} | "
        f"epochs={cfg.get('num_epochs')}\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Accuracy | {metrics['accuracy']:.4f} |\n"
        f"| Macro F1 | {metrics['macro_f1']:.4f} |\n"
        f"| Macro AUC | {metrics['macro_roc_auc']:.4f} |\n"
        f"| Tumor P / R / F1 | "
        f"{t.get('precision', 0):.3f} / {t.get('recall', 0):.3f} / {t.get('f1', 0):.3f} |\n"
        f"| Normal P / R / F1 | "
        f"{n.get('precision', 0):.3f} / {n.get('recall', 0):.3f} / {n.get('f1', 0):.3f} |\n"
    )

    with RUNS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    print(f"[runs] Appended fold {donor} → {RUNS_LOG}")


def write_summary_row(
    donor: str, metrics: dict, n_train: int, n_val: int,
    best_ckpt: str, summary_path: Path,
) -> None:
    pc = metrics.get("per_class", {})
    t = pc.get("Tumor", {})
    n = pc.get("Normal", {})
    row = {
        "fold_donor": donor,
        "n_train_cells": n_train,
        "n_val_cells": n_val,
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "macro_roc_auc": round(metrics["macro_roc_auc"], 4),
        "tumor_precision": round(t.get("precision", 0), 4),
        "tumor_recall": round(t.get("recall", 0), 4),
        "tumor_f1": round(t.get("f1", 0), 4),
        "normal_precision": round(n.get("precision", 0), 4),
        "normal_recall": round(n.get("recall", 0), 4),
        "normal_f1": round(n.get("f1", 0), 4),
        "best_checkpoint": best_ckpt,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        existing = pd.read_csv(summary_path, sep="\t")
        existing = existing[existing["fold_donor"] != donor]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated[SUMMARY_COLS].to_csv(summary_path, sep="\t", index=False)
    print(f"[summary] {donor} → {summary_path}")


# ---------------------------------------------------------------------------
# Per-fold training and evaluation
# ---------------------------------------------------------------------------

def run_fold(
    donor: str,
    tokens: np.ndarray,
    metadata: pd.DataFrame,
    encoded_labels: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    remap: Optional[np.ndarray],
    pad_fill_value: int,
    cfg: dict,
    dry_run: bool = False,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}\nLODO fold — held-out donor: {donor}\n{sep}")

    train_idx, val_idx = build_fold_indices(metadata, donor)
    n_train, n_val = len(train_idx), len(val_idx)
    train_donors = sorted(metadata.loc[train_idx, "Patient"].unique())
    print(f"Train: {n_train:,} cells from {train_donors}")
    print(f"Val:   {n_val:,} cells from [{donor}]")

    num_labels = len(id2label)

    train_ds = RankedGeneDataset(
        tokens, train_idx, encoded_labels, pad_fill_value,
        cfg["max_length"], remap,
        token_mask_prob=float(cfg.get("token_mask_prob", 0.0)),
        mask_token_id=None,
        mixup_prob=float(cfg.get("mixup_prob", 0.0)),
        mixup_alpha=float(cfg.get("mixup_alpha", 0.4)),
        num_labels=num_labels,
        rng_seed=cfg["seed"],
    )
    val_ds = RankedGeneDataset(
        tokens, val_idx, encoded_labels, pad_fill_value,
        cfg["max_length"], remap,
        token_mask_prob=0.0,
        mixup_prob=0.0,
        num_labels=num_labels,
    )
    collator = RankedGeneCollator(num_labels=num_labels)

    # --- Dry-run: check pipeline and exit ---
    if dry_run:
        batch = collator([train_ds[0], train_ds[1]])
        print(f"\n[dry-run] Donor {donor} — pipeline OK:")
        print(f"  input_ids shape : {tuple(batch['input_ids'].shape)}")
        print(f"  attention_mask  : {tuple(batch['attention_mask'].shape)}")
        print(f"  labels          : {batch['labels'].tolist()}")
        print("[dry-run] Skipping training.\n")
        return

    # --- Full training ---
    fold_out = Path(cfg["output_dir"]) / f"fold_{donor}"
    fold_out.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    sampler_weights = build_sample_weights(
        train_idx, metadata, cfg["label_column"], cfg["balance_strategy"]
    )
    if sampler_weights is not None:
        print(
            f"[sampling] class_donor — "
            f"mean={sampler_weights.mean():.3f}, std={sampler_weights.std():.3f}"
        )

    # Always fresh weights — never a prior fine-tuned checkpoint
    model_config = AutoConfig.from_pretrained(
        str(cfg["model_name_or_path"]),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg["model_name_or_path"]),
        config=model_config,
    )

    class_weights = resolve_class_weights(cfg.get("class_weights"), label2id)

    training_args = TrainingArguments(
        output_dir=str(fold_out),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=int(cfg["seed"]),
        dataloader_num_workers=0,       # required for stable fork behaviour on macOS
        dataloader_pin_memory=False,    # pin_memory unsupported on MPS
        gradient_checkpointing=False,
    )

    trainer = RankedGeneTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_f1_only,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["patience"]))],
        class_weights=class_weights,
        sampler_weights=sampler_weights,
        focal_gamma=float(cfg.get("focal_gamma") or 0.0),
    )

    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint or str(fold_out)
    print(f"\nBest checkpoint: {best_ckpt}")
    (fold_out / "best_checkpoint.txt").write_text(best_ckpt)

    # Full per-class evaluation on the held-out donor
    pred_output = trainer.predict(val_ds)
    metrics = compute_detailed_metrics(
        pred_output.predictions, pred_output.label_ids, id2label
    )
    metrics.update({
        "n_train_cells": n_train,
        "n_val_cells": n_val,
        "held_out_donor": donor,
        "best_checkpoint": best_ckpt,
        "config_snapshot": {
            k: cfg[k]
            for k in (
                "focal_gamma", "warmup_ratio", "token_mask_prob",
                "num_epochs", "learning_rate", "balance_strategy",
            )
        },
    })

    # Report against per-fold targets
    pc = metrics["per_class"]
    tumor_r = pc.get("Tumor", {}).get("recall", 0.0)
    normal_r = pc.get("Normal", {}).get("recall", 0.0)
    mf1 = metrics["macro_f1"]
    passed = mf1 > 0.65 and tumor_r > 0.70 and normal_r > 0.65
    print(
        f"\nTargets (macro_f1>0.65 AND tumor_recall>0.70 AND normal_recall>0.65): "
        f"{'PASS' if passed else 'MISS'}"
    )
    print(f"  macro_f1={mf1:.4f}  tumor_recall={tumor_r:.4f}  normal_recall={normal_r:.4f}")

    metrics_path = results_dir / f"fold_{donor}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {metrics_path}")

    write_summary_row(donor, metrics, n_train, n_val, best_ckpt,
                      results_dir / "lodo_summary.tsv")
    append_runs_log(donor, metrics, cfg, n_train, n_val)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg["seed"])

    donors = args.donors or cfg.get("donors") or ALL_DONORS
    bad = [d for d in donors if d not in ALL_DONORS]
    if bad:
        raise ValueError(f"Unknown donors: {bad}. Valid: {ALL_DONORS}")

    # Load all shared artifacts once — avoid reloading per fold
    tokens_dir = Path(cfg["tokens_dir"])
    tokens_npz = np.load(find_single(tokens_dir, TOKEN_GLOB, "token npz"))
    tokens = tokens_npz["tokens"]

    if tokens.shape[1] < int(cfg["max_length"]):
        raise ValueError(
            f"max_length={cfg['max_length']} exceeds token columns {tokens.shape[1]}."
        )

    metadata = pd.read_csv(find_single(tokens_dir, META_GLOB, "metadata"), sep="\t")
    metadata = ensure_label_column(metadata, cfg["label_column"])
    encoded_labels, id2label, label2id = prepare_label_mappings(metadata[cfg["label_column"]])

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
            raise ValueError(
                "Model config has no pad_token_id — set one explicitly in the model config."
            )
        remap = build_vocab_remap(
            dataset_vocab=dataset_vocab,
            model_vocab=model_vocab,
            pad_fill_value=dataset_pad_fill_value,
            pad_token_id=model_pad,
            unknown_token_id=model_pad,
            gene_name_map=gene_name_dict,
        )

    print(f"\nTokens : {tokens.shape}")
    print(f"Labels : {label2id}")
    print(f"Donors : {donors}")
    if args.dry_run:
        print("[dry-run] Will validate data pipeline only.\n")

    for donor in donors:
        run_fold(
            donor=donor,
            tokens=tokens,
            metadata=metadata,
            encoded_labels=encoded_labels,
            id2label=id2label,
            label2id=label2id,
            remap=remap,
            pad_fill_value=pad_fill_value,
            cfg=cfg,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print(
            f"\nAll folds complete. "
            f"Summary → {Path(cfg['results_dir']) / 'lodo_summary.tsv'}"
        )


if __name__ == "__main__":
    main()
