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
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

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
from torch.utils.data import DataLoader, Sampler
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

# Do NOT raise the MPS memory watermark. Setting PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# disables the soft limit and an over-allocation surfaces as an asynchronous Metal
# "command buffer ignored" cascade instead of a clean Python OOM. Keep the default.

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
    # Speed knobs (all metric-preserving)
    "amp_dtype": "bf16",          # "bf16" | "fp16" | "none" — autocast on MPS/CUDA
    "attn_implementation": "sdpa", # "sdpa" | "eager"
    "length_bucketing": True,      # group same-length cells per batch; needs class_donor sampler
    "bucket_mega_factor": 50,      # mega-batch = train_bs × this; sort within each
    "pad_to_multiple_of": 8,       # pad dynamic batches up to this multiple
    "save_total_limit": 2,
    "eval_accumulation_steps": 16, # stream eval logits off MPS to CPU
    "gradient_checkpointing": True,# required at L=2048 on 32GB MPS — SDPA on MPS isn't FA
}


# ---------------------------------------------------------------------------
# Speed-optimised sampler, collator, and trainer (local to LODO)
# ---------------------------------------------------------------------------

class LengthBucketedWeightedSampler(Sampler[int]):
    """Weighted draws + intra-mega-batch length sort.

    Per epoch: draw N weighted indices with replacement, split into mega-batches
    of ``batch_size * mega_factor``, sort each mega-batch by length descending,
    then flatten. The DataLoader's default batching (consecutive ``batch_size``
    indices) then produces same-length batches.

    Class-donor balancing is preserved because the weighted draw happens first;
    bucketing only re-orders the drawn indices.
    """

    def __init__(
        self,
        weights: np.ndarray,
        lengths: np.ndarray,
        batch_size: int,
        mega_factor: int = 50,
        seed: int = 0,
    ) -> None:
        if len(weights) != len(lengths):
            raise ValueError("weights and lengths must have the same length")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.mega_size = int(batch_size) * int(mega_factor)
        self.num_samples = len(weights)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        drawn = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=g
        ).numpy()
        # Shuffle so mega-batch composition varies across epochs
        drawn = drawn[rng.permutation(len(drawn))]

        bs = self.batch_size
        result: list[int] = []
        for start in range(0, len(drawn), self.mega_size):
            chunk = drawn[start:start + self.mega_size]
            # Sort the chunk by length so adjacent positions become same-length batches.
            order = np.argsort(-self.lengths[chunk], kind="stable")
            sorted_chunk = chunk[order]
            # Carve into mini-batches, then shuffle batch order. This preserves the
            # bucketing benefit (cells *within* a batch are similar length) while
            # randomising *which* batch the optimizer sees first — avoids the
            # "first iter is worst-case 2048" sawtooth and gives smoother optim dynamics.
            n_full = len(sorted_chunk) // bs
            batches = [sorted_chunk[i * bs:(i + 1) * bs] for i in range(n_full)]
            tail = sorted_chunk[n_full * bs:]
            batch_order = rng.permutation(len(batches))
            for bi in batch_order:
                result.extend(batches[bi].tolist())
            if len(tail) > 0:
                result.extend(tail.tolist())
        return iter(result)


class LengthSortedSampler(Sampler[int]):
    """Deterministic length-sorted (descending) sampler for eval."""

    def __init__(self, lengths: np.ndarray) -> None:
        self.order = np.argsort(-np.asarray(lengths), kind="stable").tolist()

    def __len__(self) -> int:
        return len(self.order)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)


class DynamicPadCollator(RankedGeneCollator):
    """Trims each batch tensor to the longest real sequence in the batch."""

    def __init__(self, num_labels: int, max_length: int, pad_to_multiple_of: int = 8):
        super().__init__(num_labels=num_labels)
        self.max_length = int(max_length)
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))

    def __call__(self, batch):
        features = super().__call__(batch)
        attn = features["attention_mask"]
        # Ranked-gene tokenization puts real tokens at the front — sum gives length.
        real_max = int(attn.sum(dim=1).max().item())
        if real_max <= 0:
            return features
        m = self.pad_to_multiple_of
        target = min(self.max_length, ((real_max + m - 1) // m) * m)
        if target < attn.shape[1]:
            features["input_ids"] = features["input_ids"][:, :target].contiguous()
            features["attention_mask"] = features["attention_mask"][:, :target].contiguous()
        return features


class FastLodoTrainer(RankedGeneTrainer):
    """RankedGeneTrainer + length-bucketed sampling + MPS autocast."""

    def __init__(
        self,
        *args,
        train_lengths: Optional[np.ndarray] = None,
        eval_lengths: Optional[np.ndarray] = None,
        bucket_mega_factor: int = 50,
        amp_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_lengths = train_lengths
        self.eval_lengths = eval_lengths
        self.bucket_mega_factor = int(bucket_mega_factor)
        self.amp_dtype = amp_dtype
        dev = self.args.device.type if hasattr(self.args, "device") else "cpu"
        self._amp_device = dev if dev in ("mps", "cuda") else None

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.sampler_weights is None or self.train_lengths is None:
            return super().get_train_dataloader()
        sampler = LengthBucketedWeightedSampler(
            weights=self.sampler_weights,
            lengths=self.train_lengths,
            batch_size=self.args.train_batch_size,
            mega_factor=self.bucket_mega_factor,
            seed=int(self.args.seed),
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=getattr(self.args, "dataloader_persistent_workers", False),
        )

    def _get_eval_sampler(self, eval_dataset):
        if self.eval_lengths is not None and len(self.eval_lengths) == len(eval_dataset):
            return LengthSortedSampler(self.eval_lengths)
        return super()._get_eval_sampler(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.amp_dtype is None or self._amp_device is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        with torch.autocast(device_type=self._amp_device, dtype=self.amp_dtype):
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        if return_outputs:
            loss, outputs = result
            return loss.float(), outputs
        return result.float()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.amp_dtype is None or self._amp_device is None:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
        with torch.autocast(device_type=self._amp_device, dtype=self.amp_dtype):
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

    def evaluate(self, *args, **kwargs):
        # Release MPS activations from the last training pass before eval allocates
        # its own working set. Prevents OOM when train and eval working sets stack.
        if self._amp_device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        result = super().evaluate(*args, **kwargs)
        if self._amp_device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return result


def _resolve_amp_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    key = str(name).lower()
    if key in ("none", "off", "false", ""):
        return None
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unknown amp_dtype: {name!r} (expected bf16, fp16, or none)")


def _compute_cell_lengths(tokens: np.ndarray, pad_marker: int = -1) -> np.ndarray:
    """Number of real tokens per cell (column count where token != pad_marker)."""
    return (tokens != pad_marker).sum(axis=1).astype(np.int64)


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
    cell_lengths: np.ndarray,
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
    collator = DynamicPadCollator(
        num_labels=num_labels,
        max_length=int(cfg["max_length"]),
        pad_to_multiple_of=int(cfg.get("pad_to_multiple_of", 8)),
    )

    # Per-position lengths aligned to train_idx / val_idx ordering
    train_lengths = cell_lengths[train_idx]
    val_lengths = cell_lengths[val_idx]
    print(
        f"[lengths] train: min={train_lengths.min()} med={int(np.median(train_lengths))} "
        f"max={train_lengths.max()} | val: min={val_lengths.min()} "
        f"med={int(np.median(val_lengths))} max={val_lengths.max()}"
    )

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
    attn_impl = str(cfg.get("attn_implementation", "sdpa") or "eager")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(cfg["model_name_or_path"]),
            config=model_config,
            attn_implementation=attn_impl,
        )
        print(f"[model] loaded with attn_implementation={attn_impl}")
    except (TypeError, ValueError) as exc:
        # Older HF or unsupported impl — fall back silently
        print(f"[model] attn_implementation={attn_impl} unsupported ({exc}); using default.")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(cfg["model_name_or_path"]),
            config=model_config,
        )

    class_weights = resolve_class_weights(cfg.get("class_weights"), label2id)

    training_args = TrainingArguments(
        output_dir=str(fold_out),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        eval_accumulation_steps=int(cfg.get("eval_accumulation_steps", 16)),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=int(cfg["seed"]),
        dataloader_num_workers=0,       # required for stable fork behaviour on macOS
        dataloader_pin_memory=False,    # pin_memory unsupported on MPS
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if training_args.gradient_checkpointing:
        print("[memory] gradient_checkpointing=True (use_reentrant=False)")

    amp_dtype = _resolve_amp_dtype(cfg.get("amp_dtype"))
    if amp_dtype is not None:
        print(f"[amp] autocast enabled — dtype={str(amp_dtype).split('.')[-1]}")

    use_bucketing = bool(cfg.get("length_bucketing", True)) and sampler_weights is not None
    trainer_cls = FastLodoTrainer
    trainer_kwargs = dict(
        train_lengths=train_lengths if use_bucketing else None,
        eval_lengths=val_lengths,
        bucket_mega_factor=int(cfg.get("bucket_mega_factor", 50)),
        amp_dtype=amp_dtype,
    )
    if use_bucketing:
        print(
            f"[sampling] length-bucketed weighted sampler "
            f"(mega_factor={trainer_kwargs['bucket_mega_factor']})"
        )

    trainer = trainer_cls(
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
        **trainer_kwargs,
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

    # Cell-level real-token counts — used for length-bucketed sampling and
    # for the dynamic-pad collator's logging. Compute once over the whole dataset.
    if "lengths" in tokens_npz.files:
        cell_lengths = tokens_npz["lengths"].astype(np.int64)
        cell_lengths = np.minimum(cell_lengths, int(cfg["max_length"]))
    else:
        cell_lengths = np.minimum(
            _compute_cell_lengths(tokens), int(cfg["max_length"])
        )
    print(
        f"Cell lengths — min={cell_lengths.min()} median={int(np.median(cell_lengths))} "
        f"mean={int(cell_lengths.mean())} max={cell_lengths.max()}"
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
            cell_lengths=cell_lengths,
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
