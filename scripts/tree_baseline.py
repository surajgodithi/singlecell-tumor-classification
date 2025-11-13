#!/usr/bin/env python
"""
Train a tree-based classifier (Random Forest or HistGradientBoosting) on the
ranked gene tokens to establish a stronger classical baseline than Naive Bayes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DEFAULT_TOKENS_DIR = Path("gse144735/processed/tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tree-based baseline (RandomForest / HistGradientBoosting) on ranked gene tokens."
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=DEFAULT_TOKENS_DIR,
        help="Directory with gene vocab, tokens NPZ, metadata TSV, and splits_by_patient.npz.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="BinaryClass",
        help="Metadata column to predict (default: BinaryClass).",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "hist_gb"],
        default="random_forest",
        help="Tree model to train (default: random_forest).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees/iterations (default: 500).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum tree depth.",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=2000,
        help="Number of most frequent genes to keep as dense features (default: 2000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model training.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU workers for RandomForest (ignored for HistGradientBoosting).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("baselines/tree_baseline_metrics.json"),
        help="Where to write the evaluation metrics JSON.",
    )
    return parser.parse_args()


def load_rank_tokens(tokens_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(tokens_path)
    return arr["tokens"], arr["lengths"]


def select_top_genes(tokens: np.ndarray, top_k: int) -> np.ndarray:
    flat = tokens.reshape(-1)
    flat = flat[flat >= 0]
    counts = np.bincount(flat)
    if len(counts) <= top_k:
        order = np.argsort(counts)[::-1]
        return order
    order = np.argpartition(counts, -top_k)[-top_k:]
    order = order[np.argsort(counts[order])[::-1]]
    return order


def build_dense_features(
    tokens: np.ndarray,
    lengths: np.ndarray,
    top_gene_ids: np.ndarray,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, Dict[int, int]]:
    n_cells = tokens.shape[0]
    feature_count = len(top_gene_ids)
    dense = np.zeros((n_cells, feature_count), dtype=dtype)
    gene_to_col = {int(gid): idx for idx, gid in enumerate(top_gene_ids)}

    for cell_idx in range(n_cells):
        cell_len = int(lengths[cell_idx])
        seq = tokens[cell_idx, :cell_len]
        # inverse rank weights as continuous inputs
        for rank, gene_id in enumerate(seq):
            if gene_id == -1:
                break
            column = gene_to_col.get(int(gene_id))
            if column is None:
                continue
            dense[cell_idx, column] += 1.0 / (rank + 1.0)
    return dense, gene_to_col


def train_model(args: argparse.Namespace, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    X_train = X[train_idx]
    y_train = y[train_idx]
    if args.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
            class_weight="balanced",
            random_state=args.random_state,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
        )
    model.fit(X_train, y_train)
    return model


def evaluate_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    class_labels: Iterable[str],
) -> Dict[str, object]:
    X_split = X[idx]
    y_true = y[idx]
    y_pred = model.predict(X_split)
    proba = model.predict_proba(X_split)

    class_ids = list(range(len(class_labels)))
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    if len(class_labels) == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        auc = roc_auc_score(y_true, proba, multi_class="ovo", average="macro")

    per_class = {}
    # compute per-class precision/recall/f1 manually for clarity
    for i, cls_name in enumerate(class_labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tp + fn),
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_auc": auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(labels)}
    mat = np.zeros((len(mapping), len(mapping)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        mat[mapping[truth], mapping[pred]] += 1
    return mat


def main() -> None:
    args = parse_args()
    tokens, lengths = load_rank_tokens(args.tokens_dir / "gse144735_gene_rank_tokens.npz")
    metadata = pd.read_csv(args.tokens_dir / "gse144735_tokens_metadata.tsv", sep="\t")

    if args.label_column not in metadata.columns:
        raise KeyError(f"Label column '{args.label_column}' not found in metadata.")
    label_strings = metadata[args.label_column].astype(str).to_numpy()
    class_labels = sorted(np.unique(label_strings).tolist())
    label_to_id = {cls: idx for idx, cls in enumerate(class_labels)}
    y = np.array([label_to_id[val] for val in label_strings], dtype=np.int64)

    top_gene_ids = select_top_genes(tokens, args.top_genes)
    print(f"[features] Selected top {len(top_gene_ids)} genes for dense features.")
    X, _ = build_dense_features(tokens, lengths, top_gene_ids)
    print(f"[features] Built dense matrix with shape {X.shape} (dtype={X.dtype}).")

    splits = np.load(args.tokens_dir / "splits_by_patient.npz", allow_pickle=True)
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    model = train_model(args, X, y, train_idx)
    metrics = {
        "train": evaluate_split(model, X, y, train_idx, class_labels),
        "val": evaluate_split(model, X, y, val_idx, class_labels),
        "test": evaluate_split(model, X, y, test_idx, class_labels),
        "classes": class_labels,
        "model_type": args.model_type,
        "top_gene_count": int(len(top_gene_ids)),
        "feature_strategy": {
            "top_genes": args.top_genes,
            "aggregation": "inverse_rank_sum",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
