#!/usr/bin/env python
"""
Fine-tune a pretrained single-cell transformer (e.g., Geneformer/scGPT) on the
ranked token dataset derived from GSE144735.

The script loads the NPZ tokens, donor-wise split indices, and metadata labels,
optionally remaps dataset-specific gene ids to the pretrained model's vocabulary,
and launches a Hugging Face Trainer run for sequence classification.
"""

from __future__ import annotations

import argparse
import json
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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_CONFIG_PATH = Path("configs/finetune.yaml")
PATH_FIELDS = {"tokens_dir", "model_vocab", "output_dir", "config"}
FALLBACK_DEFAULTS = {
    "tokens_dir": Path("gse144735/processed/tokens"),
    "output_dir": Path("outputs/transformer_finetune"),
    "max_length": 2048,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_epochs": 5,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "seed": 42,
    "patience": 2,
    "label_column": "Class",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer classifier on ranked gene tokens."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (default: configs/finetune.yaml when present).",
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=None,
        help="Directory containing {gene_vocab.tsv, *_gene_rank_tokens.npz, splits_by_patient.npz, *_tokens_metadata.tsv}.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Hugging Face model identifier or local checkpoint (e.g., geneformer/geneformer).",
    )
    parser.add_argument(
        "--model-vocab",
        type=Path,
        help="Optional TSV file mapping gene symbols to token ids for the pretrained model. "
        "Provide this when dataset tokens need to be realigned to the model vocabulary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum number of ranked genes per cell (should match tokenisation).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of evaluation steps with no improvement before early stopping. "
        "Set <=0 to disable.",
    )
    parser.add_argument(
        "--model-pad-token-id",
        type=int,
        help="Override the pad token id used by the pretrained model. "
        "Defaults to the model config value.",
    )
    parser.add_argument(
        "--unknown-token-id",
        type=int,
        help="Token id to use when a dataset gene is missing from the model vocabulary. "
        "Defaults to the model pad token id.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Metadata column to predict (defaults to Tumor/Border/Normal classes).",
    )
    return parser


def load_config(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    preliminary = parser.parse_known_args()[0]
    config_path: Optional[Path] = preliminary.config
    if config_path is None and DEFAULT_CONFIG_PATH.exists():
        config_path = DEFAULT_CONFIG_PATH

    config_data: Dict[str, object] = {}
    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found.")
        config_data = load_config(config_path)

    args = parser.parse_args()
    args_dict = vars(args)

    if config_data:
        print(f"Loaded config: {config_path}")
        for key, value in config_data.items():
            if key not in args_dict:
                print(f"[warn] Unknown config key '{key}' – ignoring.")
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

    if args.model_name_or_path is None:
        raise SystemExit("`model-name-or-path` must be specified via CLI or config.")

    args.config_path = str(config_path) if config_path else None
    return args


def load_gene_vocab(path: Path) -> pd.Series:
    """Load a TSV vocabulary (gene_symbol -> token_id)."""
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1:
        raise ValueError(f"{path} must contain at least two columns (gene symbol + token id).")
    symbol_col = df.columns[0]
    if "token_id" not in df.columns:
        raise ValueError(f"{path} must include a 'token_id' column.")
    series = df.set_index(symbol_col)["token_id"].astype(int)
    return series


def build_vocab_remap(
    dataset_vocab: pd.Series,
    model_vocab: pd.Series,
    pad_fill_value: int,
    pad_token_id: int,
    unknown_token_id: int,
) -> np.ndarray:
    """Create an array that maps dataset token ids to model token ids."""
    max_dataset_id = int(dataset_vocab.max())
    remap = np.full(max_dataset_id + 2, fill_value=unknown_token_id, dtype=np.int64)
    model_lookup = model_vocab.to_dict()
    missing = 0
    for gene_symbol, dataset_id in dataset_vocab.items():
        model_id = model_lookup.get(gene_symbol)
        if model_id is None:
            missing += 1
            continue
        remap[int(dataset_id)] = int(model_id)
    remap[pad_fill_value] = pad_token_id
    if missing:
        rate = missing / len(dataset_vocab)
        print(f"[warn] {missing} / {len(dataset_vocab)} genes ({rate:.1%}) missing from model vocab; mapped to unknown_token_id={unknown_token_id}.")
    return remap


class RankedGeneDataset(Dataset):
    """Torch dataset that surfaces ranked gene tokens with attention masks."""

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

    def __getitem__(self, position: int) -> Dict[str, torch.Tensor]:
        idx = self.indices[position]
        seq = self.tokens[idx, : self.max_length].astype(np.int64).copy()
        mask = seq != -1
        seq[~mask] = self.pad_fill_value
        if self.remap is not None:
            seq = self.remap[seq]
        attn_mask = mask.astype(np.int64)
        return {
            "input_ids": torch.from_numpy(seq),
            "attention_mask": torch.from_numpy(attn_mask),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }
    try:
        probs = softmax(logits)
        metrics["macro_auc"] = roc_auc_score(labels, probs, multi_class="ovo", average="macro")
    except ValueError:
        pass
    return metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def prepare_label_mappings(series: pd.Series) -> tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    labels = series.unique()
    sorted_labels = sorted(labels)
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    encoded = series.map(label2id).to_numpy(dtype=np.int64)
    return encoded, id2label, label2id


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = args.tokens_dir / "gse144735_gene_rank_tokens.npz"
    metadata_path = args.tokens_dir / "gse144735_tokens_metadata.tsv"
    splits_path = args.tokens_dir / "splits_by_patient.npz"
    dataset_vocab_path = args.tokens_dir / "gene_vocab.tsv"

    tokens_npz = np.load(tokens_path)
    tokens = tokens_npz["tokens"]
    lengths = tokens_npz["lengths"]
    if args.max_length > tokens.shape[1]:
        raise ValueError(f"max_length={args.max_length} exceeds token width {tokens.shape[1]}")

    metadata = pd.read_csv(metadata_path, sep="\t")
    if args.label_column not in metadata.columns:
        raise KeyError(f"{args.label_column} not found in metadata columns: {metadata.columns.tolist()}")
    encoded_labels, id2label, label2id = prepare_label_mappings(metadata[args.label_column])

    splits = np.load(splits_path, allow_pickle=True)
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    dataset_vocab = load_gene_vocab(dataset_vocab_path)
    dataset_pad_fill_value = int(dataset_vocab.max()) + 1

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    model_pad_token_id = (
        args.model_pad_token_id if args.model_pad_token_id is not None else config.pad_token_id
    )
    if model_pad_token_id is None:
        raise ValueError("Pad token id is undefined. Provide --model-pad-token-id or use a model with pad_token_id set.")
    unknown_token_id = args.unknown_token_id if args.unknown_token_id is not None else model_pad_token_id

    remap = None
    if args.model_vocab:
        model_vocab = load_gene_vocab(args.model_vocab)
        remap = build_vocab_remap(
            dataset_vocab=dataset_vocab,
            model_vocab=model_vocab,
            pad_fill_value=dataset_pad_fill_value,
            pad_token_id=model_pad_token_id,
            unknown_token_id=unknown_token_id,
        )
        pad_fill_value = dataset_pad_fill_value
    else:
        # Ensure vocab sizes already match when remap is omitted
        if config.vocab_size <= int(dataset_vocab.max()):
            raise ValueError(
                "Model vocab size is smaller than dataset token ids. "
                "Provide --model-vocab so tokens can be remapped."
            )
        pad_fill_value = model_pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    train_dataset = RankedGeneDataset(tokens, train_idx, encoded_labels, pad_fill_value, args.max_length, remap)
    val_dataset = RankedGeneDataset(tokens, val_idx, encoded_labels, pad_fill_value, args.max_length, remap)
    test_dataset = RankedGeneDataset(tokens, test_idx, encoded_labels, pad_fill_value, args.max_length, remap)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    metrics_payload = {
        "val": val_metrics,
        "test": test_metrics,
        "label_mapping": id2label,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    mapping_path = args.output_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps({"id2label": id2label, "label2id": label2id}, indent=2))

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
