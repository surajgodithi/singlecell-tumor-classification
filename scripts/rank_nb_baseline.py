#!/usr/bin/env python
"""
Rank-based Naive Bayes baseline for the GSE144735 token dataset.

This script reconstructs sparse features from the ranked gene tokens
(`gse144735_gene_rank_tokens.npz`), trains a simple multinomial Naive Bayes
classifier with donor-wise splits, and prints accuracy, macro AUC, and
per-class precision/recall/F1 metrics. It mirrors the sanity-check baseline
developed during notebook exploration so we can reproduce and showcase the
initial results before scaling up with Geneformer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


DEFAULT_TOKENS_DIR = Path("gse144735/processed/tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a rank-based Naive Bayes baseline on the tokenised GSE144735 dataset."
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=DEFAULT_TOKENS_DIR,
        help="Directory containing gene_vocab.tsv, gse144735_gene_rank_tokens.npz, and split artefacts.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-2,
        help="Additive smoothing factor for the Naive Bayes weights.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the final metrics dictionary as JSON.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Class",
        help="Metadata column to use as labels (default: Class).",
    )
    return parser.parse_args()


def build_rank_feature_matrix(tokens: np.ndarray, lengths: np.ndarray, vocab_size: int) -> sparse.csr_matrix:
    """
    Convert ranked gene indices into a sparse CSR matrix using inverse-rank weights.
    """
    n_cells, max_len = tokens.shape
    valid_mask = tokens != -1
    valid_tokens = tokens[valid_mask].astype(np.int32)

    row_idx = np.repeat(np.arange(n_cells, dtype=np.int32), lengths)

    # Efficiently build the rank positions without allocating Python lists of arrays.
    total_entries = int(lengths.sum())
    ranks = np.empty(total_entries, dtype=np.int32)
    cursor = 0
    for length in lengths:
        ranks[cursor : cursor + length] = np.arange(length, dtype=np.int32)
        cursor += length

    values = 1.0 / (ranks + 1.0)
    matrix = sparse.csr_matrix((values, (row_idx, valid_tokens)), shape=(n_cells, vocab_size))
    return matrix


@dataclass
class RankNaiveBayes:
    alpha: float = 1e-2
    classes_: np.ndarray | None = None
    log_theta_: np.ndarray | None = None
    log_prior_: np.ndarray | None = None

    def fit(self, X: sparse.csr_matrix, y: np.ndarray) -> "RankNaiveBayes":
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()
        classes, inverse = np.unique(y, return_inverse=True)
        self.classes_ = classes

        log_theta = []
        log_prior = []
        vocab_size = X.shape[1]

        for idx, cls in enumerate(classes):
            mask = inverse == idx
            X_cls = X[mask]
            counts = np.asarray(X_cls.sum(axis=0)).ravel()
            theta = (counts + self.alpha) / (counts.sum() + self.alpha * vocab_size)
            log_theta.append(np.log(theta))
            log_prior.append(np.log(mask.sum() / len(y)))

        self.log_theta_ = np.vstack(log_theta)
        self.log_prior_ = np.array(log_prior)
        return self

    def _raw_scores(self, X: sparse.csr_matrix) -> np.ndarray:
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()
        return X.dot(self.log_theta_.T) + self.log_prior_

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        scores = self._raw_scores(X)
        max_scores = scores.max(axis=1, keepdims=True)
        probs = np.exp(scores - max_scores)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Iterable[str]) -> np.ndarray:
    idx_map = {cls: i for i, cls in enumerate(classes)}
    mat = np.zeros((len(idx_map), len(idx_map)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        mat[idx_map[true_label], idx_map[pred_label]] += 1
    return mat


def per_class_metrics(cm: np.ndarray, classes: Iterable[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    total = cm.sum()
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(tp + fn),
        }
    return metrics


def macro_auc(y_true: np.ndarray, probs: np.ndarray, classes: Iterable[str]) -> float:
    aucs = []
    for col, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        pos = binary_true.sum()
        neg = len(binary_true) - pos
        if pos == 0 or neg == 0:
            continue
        ranks = rankdata(probs[:, col], method="average")
        pos_rank_sum = ranks[binary_true == 1].sum()
        auc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_split(
    name: str,
    model: RankNaiveBayes,
    X: sparse.csr_matrix,
    y: np.ndarray,
    idx: np.ndarray,
) -> Dict[str, object]:
    X_split = X[idx]
    y_true = y[idx]
    probs = model.predict_proba(X_split)
    y_pred = model.predict(X_split)

    cm = compute_confusion_matrix(y_true, y_pred, model.classes_)
    acc = cm.trace() / cm.sum()
    auc = macro_auc(y_true, probs, model.classes_)
    metrics = per_class_metrics(cm, model.classes_)

    print(f"\n--- {name.upper()} ---")
    print(f"Accuracy: {acc:.3f}, Macro ROC AUC: {auc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {"accuracy": acc, "macro_auc": auc, "per_class": metrics}


def main() -> None:
    args = parse_args()
    token_dir = args.tokens_dir

    tokens_npz = np.load(token_dir / "gse144735_gene_rank_tokens.npz")
    tokens = tokens_npz["tokens"]
    lengths = tokens_npz["lengths"]

    vocab = pd.read_csv(token_dir / "gene_vocab.tsv", sep="\t")
    vocab_size = vocab["token_id"].max() + 1

    print(f"Loaded tokens: shape={tokens.shape}, vocab={vocab_size}, total entries={int(lengths.sum())}")

    X = build_rank_feature_matrix(tokens, lengths, vocab_size)
    metadata = pd.read_csv(token_dir / "gse144735_tokens_metadata.tsv", sep="\t")
    splits = np.load(token_dir / "splits_by_patient.npz", allow_pickle=True)
    if args.label_column not in metadata.columns:
        available = ", ".join(metadata.columns)
        raise KeyError(f"Label column '{args.label_column}' not found in metadata. Available columns: {available}")
    y = metadata[args.label_column].to_numpy()

    print(f"Sparse matrix nnz={X.nnz} ({X.nnz / (X.shape[0] * X.shape[1]):.6f} density)")
    print(f"Split sizes: train={len(splits['train_idx'])}, val={len(splits['val_idx'])}, test={len(splits['test_idx'])}")

    model = RankNaiveBayes(alpha=args.alpha).fit(X[splits["train_idx"]], y[splits["train_idx"]])

    results = {}
    for split_name in ["train_idx", "val_idx", "test_idx"]:
        human_name = split_name.replace("_idx", "")
        results[human_name] = evaluate_split(human_name, model, X, y, splits[split_name])

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote metrics to {args.output_json}")
    else:
        print("\nFinal metrics JSON:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
