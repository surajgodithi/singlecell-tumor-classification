#!/usr/bin/env python
"""
Evaluate a fine-tuned transformer checkpoint on the donor-wise splits.

Usage example:
python scripts/evaluate_transformer.py \
  --tokens-dir gse144735/processed/tokens \
  --model-path outputs/geneformer_colon/best-model \
  --model-vocab /content/Geneformer/geneformer/token_dictionary_gc104M.pkl \
  --model-gene-name-dict /content/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

DEFAULT_CONFIG_PATH = Path("configs/eval.yaml")
PATH_FIELDS = {
    "tokens_dir",
    "model_path",
    "model_vocab",
    "model_gene_name_dict",
    "output_json",
    "config",
}
FALLBACK_DEFAULTS = {
    "tokens_dir": Path("gse144735/processed/tokens"),
    "max_length": 2048,
    "output_json": Path("outputs/eval_metrics.json"),
    "eval_splits": ["val", "test"],
    "eval_batch_size": 8,
    "label_column": "Class",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned transformer checkpoint on the donor splits."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (default: configs/eval.yaml when present).",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=None,
        help="Directory containing gene_vocab.tsv (or pickle), *_gene_rank_tokens.npz, *_tokens_metadata.tsv, splits_by_patient.npz.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the fine-tuned model checkpoint directory (e.g., outputs/.../best-model).",
    )
    parser.add_argument(
        "--model-vocab",
        type=Path,
        default=None,
        help="Path to the pretrained model's vocabulary (TSV or Geneformer pickle).",
    )
    parser.add_argument(
        "--model-gene-name-dict",
        type=Path,
        help="Optional path to Geneformer's gene_name_id_dict (pickle/TSV) for improved alias mapping.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Metadata column containing labels.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum ranked genes per cell (should equal training).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the final metrics JSON.",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Which splits to evaluate (default val and test).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size.",
    )
    return parser


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    preliminary = parser.parse_known_args()[0]
    config_path = preliminary.config or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
    config_data: Dict[str, object] = load_config(config_path) if config_path else {}

    args = parser.parse_args()
    args_dict = vars(args)

    if config_data:
        print(f"Loaded config: {config_path}")
        for key, value in config_data.items():
            if key not in args_dict:
                print(f"[warn] Unknown config key '{key}' - ignoring.")
                continue
            if args_dict[key] is None:
                args_dict[key] = value

    for key, value in FALLBACK_DEFAULTS.items():
        if args_dict.get(key) is None:
            args_dict[key] = value

    for field in PATH_FIELDS:
        value = args_dict.get(field)
        if value is not None and not isinstance(value, Path):
            args_dict[field] = Path(value)

    missing = [field for field in ["model_path", "model_vocab"] if args_dict.get(field) is None]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join(missing)}. Provide via CLI or config.")

    args.config_path = str(config_path) if config_path else None
    return args


def load_gene_vocab(path: Path) -> pd.Series:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, dict):
            raise ValueError(f"{path} must be a dict mapping gene names to token ids.")
        series = pd.Series(obj, name="token_id")
        series.index.name = "gene_symbol"
        return series.astype(int)

    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1 or "token_id" not in df.columns:
        raise ValueError(f"{path} must have at least two columns including 'token_id'.")
    symbol_col = df.columns[0]
    return df.set_index(symbol_col)["token_id"].astype(int)


def load_gene_name_dict(path: Path) -> Dict[str, str]:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as fh:
            obj = pickle.load(fh)
    else:
        df = pd.read_csv(path, sep="\t", header=None, names=["gene_symbol", "gene_id"])
        obj = dict(zip(df["gene_symbol"], df["gene_id"]))

    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a dict mapping gene symbols to gene IDs.")

    def canonical(value: str) -> str:
        return str(value).strip().upper()

    mapping: Dict[str, str] = {}
    for key, value in obj.items():
        if value is None:
            continue
        key_norm = canonical(key)
        if isinstance(value, (list, tuple, set)):
            for candidate in value:
                if candidate:
                    mapping[key_norm] = canonical(candidate)
                    break
        else:
            mapping[key_norm] = canonical(value)
    return mapping


def canonical_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def build_vocab_remap(
    dataset_vocab: pd.Series,
    model_vocab: pd.Series,
    pad_fill_value: int,
    pad_token_id: int,
    unknown_token_id: int,
    gene_name_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    max_dataset_id = int(dataset_vocab.max())
    remap = np.full(max_dataset_id + 2, fill_value=unknown_token_id, dtype=np.int64)

    model_lookup = {canonical_symbol(sym): int(tok) for sym, tok in model_vocab.items()}
    alias_lookup = {canonical_symbol(k): canonical_symbol(v) for k, v in (gene_name_map or {}).items()}

    direct_matches = 0
    alias_matches = 0
    missing = 0

    for gene_symbol, dataset_id in dataset_vocab.items():
        key = canonical_symbol(gene_symbol)
        model_id = model_lookup.get(key)
        if model_id is None:
            alias = alias_lookup.get(key)
            if alias:
                model_id = model_lookup.get(alias)
                if model_id is not None:
                    alias_matches += 1
        else:
            direct_matches += 1

        if model_id is None:
            missing += 1
            continue
        remap[int(dataset_id)] = int(model_id)

    total = len(dataset_vocab)
    if total:
        matched = direct_matches + alias_matches
        print(
            f"[remap] matched {matched} genes ({direct_matches} direct, {alias_matches} via alias map); "
            f"missing {missing} ({missing/total:.1%})."
        )
    remap[pad_fill_value] = pad_token_id
    return remap


class RankedGeneDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        indices: np.ndarray,
        labels: np.ndarray,
        pad_fill_value: int,
        max_length: int,
        remap: Optional[np.ndarray] = None,
    ) -> None:
        self.tokens = tokens
        self.indices = indices
        self.labels = labels
        self.pad_fill_value = pad_fill_value
        self.max_length = max_length
        self.remap = remap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_idx = self.indices[idx]
        seq = self.tokens[token_idx, : self.max_length].astype(np.int64).copy()
        mask = seq != -1
        seq[~mask] = self.pad_fill_value
        if self.remap is not None:
            seq = self.remap[seq]
        attn_mask = mask.astype(np.int64)
        return {
            "input_ids": torch.from_numpy(seq),
            "attention_mask": torch.from_numpy(attn_mask),
            "labels": torch.tensor(self.labels[token_idx], dtype=torch.long),
        }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: Iterable[str]) -> np.ndarray:
    idx_map = {cls: i for i, cls in enumerate(classes)}
    mat = np.zeros((len(idx_map), len(idx_map)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        mat[idx_map[true_label], idx_map[pred_label]] += 1
    return mat


def per_class_metrics(cm: np.ndarray, classes: Iterable[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
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
        ranks = np.argsort(np.argsort(probs[:, col])) + 1
        pos_rank_sum = ranks[binary_true == 1].sum()
        auc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def prepare_label_mappings(series: pd.Series) -> tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    labels = sorted(series.unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    encoded = series.map(label2id).to_numpy(dtype=np.int64)
    return encoded, id2label, label2id


def align_labels_with_config(series: pd.Series, config) -> tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    if not config.label2id:
        return prepare_label_mappings(series)

    label2id = {canonical_symbol(k): int(v) for k, v in config.label2id.items()}
    id2label = {int(v): k for k, v in config.label2id.items()}
    encoded = series.map(lambda x: label2id[canonical_symbol(x)]).to_numpy(dtype=np.int64)
    readable_id2label = {idx: label for idx, label in id2label.items()}
    readable_label2id = {label: idx for idx, label in readable_id2label.items()}
    return encoded, readable_id2label, readable_label2id


def evaluate_split(trainer: Trainer, dataset: Dataset, human_name: str, classes: Iterable[str]) -> Dict[str, object]:
    prediction_output = trainer.predict(dataset)
    logits = prediction_output.predictions
    preds = np.argmax(logits, axis=1)
    labels = prediction_output.label_ids

    probs = softmax(logits)
    id2label = {idx: label for idx, label in enumerate(classes)}
    label_strings = np.array([id2label[idx] for idx in labels])
    pred_strings = np.array([id2label[idx] for idx in preds])

    cm = compute_confusion_matrix(label_strings, pred_strings, classes)
    accuracy = accuracy_score(label_strings, pred_strings)
    macro_f1 = f1_score(label_strings, pred_strings, average="macro")
    auc = macro_auc(label_strings, probs, classes)
    per_class = per_class_metrics(cm, classes)

    print(f"\n--- {human_name.upper()} ---")
    print(f"Accuracy: {accuracy:.3f}, Macro F1: {macro_f1:.3f}, Macro ROC AUC: {auc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_auc": auc,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def resolve_model_path(model_path: Path) -> Path:
    """If model_path is a directory containing trainer_state.json, return the best checkpoint path."""
    if (model_path / "trainer_state.json").exists():
        state = json.loads((model_path / "trainer_state.json").read_text())
        best_ckpt = state.get("best_model_checkpoint")
        if best_ckpt:
            candidate = Path(best_ckpt)
            if candidate.exists():
                print(f"[eval] Using best checkpoint from trainer_state.json: {candidate}")
                return candidate
            # try relative to the parent directory if the stored path was relative
            relative_candidate = (model_path / candidate).resolve()
            if relative_candidate.exists():
                print(f"[eval] Using best checkpoint (relative) from trainer_state.json: {relative_candidate}")
                    return relative_candidate
            print(f"[warn] Stored best checkpoint path {best_ckpt} not found; falling back to model_path.")
    if model_path.is_dir():
        config_file = model_path / "config.json"
        if config_file.exists():
            return model_path
        def checkpoint_order(p: Path) -> int:
            tail = p.name.split("-")[-1]
            return int(tail) if tail.isdigit() else -1

        checkpoints = sorted((p for p in model_path.glob("checkpoint-*") if p.is_dir()), key=checkpoint_order)
        if checkpoints:
            chosen = checkpoints[-1]
            print(f"[eval] No trainer_state.json found; selecting latest checkpoint: {chosen}")
            return chosen
        raise FileNotFoundError(
            f"No config.json or checkpoint-* directories found under {model_path}. "
            "Please point model_path to a specific checkpoint."
        )
    return model_path


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    tokens_npz = np.load(args.tokens_dir / "gse144735_gene_rank_tokens.npz")
    tokens = tokens_npz["tokens"]
    lengths = tokens_npz["lengths"]
    if args.max_length > tokens.shape[1]:
        raise ValueError(f"max_length={args.max_length} exceeds token width {tokens.shape[1]}")

    metadata = pd.read_csv(args.tokens_dir / "gse144735_tokens_metadata.tsv", sep="\t")
    splits = np.load(args.tokens_dir / "splits_by_patient.npz", allow_pickle=True)

    dataset_vocab = load_gene_vocab(args.tokens_dir / "gene_vocab.tsv")
    dataset_pad_fill_value = int(dataset_vocab.max()) + 1

    model_vocab = load_gene_vocab(args.model_vocab)
    gene_name_dict = load_gene_name_dict(args.model_gene_name_dict) if args.model_gene_name_dict else None

    config = AutoConfig.from_pretrained(str(model_path))
    labels_encoded, id2label_map, label2id_map = align_labels_with_config(metadata[args.label_column], config)
    classes = [id2label_map[idx] for idx in sorted(id2label_map)]

    pad_token_id = config.pad_token_id
    if pad_token_id is None:
        raise ValueError("Model config must define pad_token_id.")
    unknown_token_id = pad_token_id

    remap = build_vocab_remap(
        dataset_vocab=dataset_vocab,
        model_vocab=model_vocab,
        pad_fill_value=dataset_pad_fill_value,
        pad_token_id=pad_token_id,
        unknown_token_id=unknown_token_id,
        gene_name_map=gene_name_dict,
    )
    pad_fill_value = dataset_pad_fill_value

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), config=config)
    training_args = TrainingArguments(
        output_dir=str(args.model_path / "eval_tmp"),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_drop_last=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args)

    results: Dict[str, Dict[str, object]] = {}
    for split_name in args.eval_splits:
        idx_key = f"{split_name}_idx"
        if idx_key not in splits:
            raise KeyError(f"{idx_key} not found in splits file.")
        dataset = RankedGeneDataset(
            tokens,
            splits[idx_key],
            labels_encoded,
            pad_fill_value,
            args.max_length,
            remap,
        )
        results[split_name] = evaluate_split(trainer, dataset, split_name, classes)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote evaluation metrics to {args.output_json}")


if __name__ == "__main__":
    main()
