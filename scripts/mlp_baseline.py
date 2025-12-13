#!/usr/bin/env python
"""
Shallow MLP baseline on ranked gene tokens (Binary or multi-class labels).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DEFAULT_TOKENS_DIR = Path("gse144735/processed/tokens")
TOKEN_GLOB = "*_gene_rank_tokens.npz"
META_GLOB = "*_tokens_metadata.tsv"


def ensure_label_column(metadata: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if label_column in metadata.columns:
        return metadata
    if label_column == "BinaryClass" and "Class" in metadata.columns:
        metadata = metadata.copy()
        metadata[label_column] = metadata["Class"].replace({"Border": "Normal"})
        return metadata
    raise KeyError(f"Label column '{label_column}' not found in metadata columns: {metadata.columns.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shallow MLP baseline on ranked gene tokens.")
    parser.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR, help="Directory with tokens + metadata.")
    parser.add_argument("--label-column", type=str, default="BinaryClass", help="Metadata column to predict.")
    parser.add_argument("--top-genes", type=int, default=2000, help="Number of most frequent genes to keep.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,256",
        help="Comma-separated hidden layer sizes (default: 512,256).",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda|cpu|auto).")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("baselines/gse144735_mlp_binary_metrics.json"),
        help="Where to save metrics JSON.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_tokens(tokens_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(tokens_path)
    return arr["tokens"], arr["lengths"]


def select_top_genes(tokens: np.ndarray, top_k: int) -> np.ndarray:
    flat = tokens.reshape(-1)
    flat = flat[flat >= 0]
    counts = np.bincount(flat)
    if len(counts) <= top_k:
        return np.argsort(counts)[::-1]
    idx = np.argpartition(counts, -top_k)[-top_k:]
    return idx[np.argsort(counts[idx])[::-1]]


def build_dense_features(tokens: np.ndarray, lengths: np.ndarray, top_gene_ids: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    n_cells = tokens.shape[0]
    feature_count = len(top_gene_ids)
    dense = np.zeros((n_cells, feature_count), dtype=np.float32)
    gene_to_col = {int(g): idx for idx, g in enumerate(top_gene_ids)}
    for cell_idx in range(n_cells):
        cell_len = int(lengths[cell_idx])
        seq = tokens[cell_idx, :cell_len]
        for rank, gene_id in enumerate(seq):
            if gene_id == -1:
                break
            col = gene_to_col.get(int(gene_id))
            if col is None:
                continue
            dense[cell_idx, col] += 1.0 / (rank + 1.0)
    return dense, gene_to_col


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def standardize(train_X: np.ndarray, others: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True) + 1e-6
    train_std = (train_X - mean) / std
    transformed = [(arr - mean) / std for arr in others]
    return train_std, transformed


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, class_names: List[str]) -> Dict[str, object]:
    num_classes = len(class_names)
    preds = logits.argmax(axis=1)
    probs = softmax_np(logits)
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    if num_classes == 2:
        auc = roc_auc_score(y_true, probs[:, 1])
    else:
        auc = roc_auc_score(y_true, probs, multi_class="ovo", average="macro")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, preds):
        cm[true, pred] += 1

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(tp + fn)}

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_auc": auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(yb.numpy())
    return np.concatenate(labels_list), np.concatenate(logits_list)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)

    token_files = list(args.tokens_dir.glob(TOKEN_GLOB))
    if not token_files:
        raise FileNotFoundError(f"No token file matching {TOKEN_GLOB} found under {args.tokens_dir}")
    if len(token_files) > 1:
        raise FileExistsError(f"Multiple token files found under {args.tokens_dir}: {token_files}")
    tokens, lengths = load_tokens(token_files[0])

    metadata_files = list(args.tokens_dir.glob(META_GLOB))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file matching {META_GLOB} found under {args.tokens_dir}")
    if len(metadata_files) > 1:
        raise FileExistsError(f"Multiple metadata files found under {args.tokens_dir}: {metadata_files}")
    metadata = pd.read_csv(metadata_files[0], sep="\t")
    metadata = ensure_label_column(metadata, args.label_column)
    labels = metadata[args.label_column].astype(str).to_numpy()
    class_names = sorted(np.unique(labels).tolist())
    label_to_id = {cls: idx for idx, cls in enumerate(class_names)}
    y = np.array([label_to_id[val] for val in labels], dtype=np.int64)

    top_gene_ids = select_top_genes(tokens, args.top_genes)
    X, _ = build_dense_features(tokens, lengths, top_gene_ids)

    splits = np.load(args.tokens_dir / "splits_by_patient.npz", allow_pickle=True)
    train_idx, val_idx, test_idx = splits["train_idx"], splits["val_idx"], splits["test_idx"]

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    X_train, transformed = standardize(X_train, [X_val, X_test])
    X_val, X_test = transformed

    train_dataset = ArrayDataset(X_train, y[train_idx])
    val_dataset = ArrayDataset(X_val, y[val_idx])
    test_dataset = ArrayDataset(X_test, y[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
    model = MLP(X_train.shape[1], hidden_dims, len(class_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_f1 = -np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        val_labels, val_logits = evaluate(model, val_loader, device)
        metrics = compute_metrics(val_labels, val_logits, class_names)
        val_f1 = metrics["macro_f1"]
        print(f"Epoch {epoch}: val macro F1={val_f1:.3f}, acc={metrics['accuracy']:.3f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    results = {}
    for split_name, loader in [
        ("train", train_eval_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        labels_arr, logits_arr = evaluate(model, loader, device)
        results[split_name] = compute_metrics(labels_arr, logits_arr, class_names)

    payload = {
        "top_gene_count": int(len(top_gene_ids)),
        "feature_strategy": {"top_genes": args.top_genes, "aggregation": "inverse_rank_sum"},
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "metrics": results,
        "classes": class_names,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
