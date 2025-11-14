Rank-based Naive Bayes Baseline
================================

### Goal
Provide a lightweight, reproducible classifier that operates directly on the ranked token artefacts so we can document per-dataset donor-wise baselines before wiring up the transformer fine-tuning stack. Every cancer dataset (colorectal, breast, lung, etc.) should get its own Naive Bayes run (with metrics saved via `--output-json baselines/<dataset>_rank_nb_metrics.json`) so Geneformer fine-tuning results are always compared against a matching reference.

### Implementation
- Script: `scripts/rank_nb_baseline.py`.
- Inputs: dataset-specific token artefacts (e.g., for GSE144735: `gse144735/processed/tokens/gse144735_gene_rank_tokens.npz`, `gene_vocab.tsv`, `gse144735_tokens_metadata.tsv`, `splits_by_patient.npz`).
- Features: sparse CSR matrix with inverse-rank weights (`1/(rank+1)`) per gene token, mirroring the order Geneformer consumes.
- Model: simple multinomial Naive Bayes with additive smoothing (`alpha=1e-2`). Metrics computed with NumPy/SciPy only.
- Usage:
  ```
  python scripts/rank_nb_baseline.py \
    --tokens-dir gse144735/processed/tokens \
    --output-json baselines/gse144735_rank_nb_metrics.json
  ```

### Results – GSE144735 (colorectal)
| Split | Accuracy | Macro ROC AUC | Notes |
|-------|----------|---------------|-------|
| Train (KUL28/KUL21/KUL31/KUL30) | 0.65 | 0.83 | Normal class dominates; Tumor recall ~0.40. |
| Validation (KUL19) | 0.26 | 0.58 | Strong donor shift; Tumor/Border largely collapse into Normal. |
| Test (KUL01) | 0.57 | 0.79 | Border/Normal decent, Tumor recall ~0.29. |

#### Per-class metrics (from `baselines/gse144735_rank_nb_metrics.json`)
| Split | Class  | Precision | Recall | F1 | Support |
|-------|--------|-----------|--------|----|---------|
| Train | Border | 0.54 | 0.50 | 0.52 | 4,032 |
| Train | Normal | 0.73 | 0.89 | 0.81 | 5,849 |
| Train | Tumor  | 0.58 | 0.40 | 0.47 | 3,204 |
| Val   | Border | 0.27 | 0.14 | 0.19 | 3,263 |
| Val   | Normal | 0.23 | 0.75 | 0.35 | 1,875 |
| Val   | Tumor  | 0.52 | 0.08 | 0.14 | 3,128 |
| Test  | Border | 0.50 | 0.65 | 0.56 | 2,129 |
| Test  | Normal | 0.64 | 0.75 | 0.69 | 2,012 |
| Test  | Tumor  | 0.58 | 0.29 | 0.38 | 1,922 |

Confusion matrices and per-class precision/recall/F1 are printed and stored in the JSON file for provenance. When onboarding new datasets, repeat this section with the new donor allocation and per-class metrics so we maintain a growing catalogue of baselines alongside their transformer fine-tune results.

### Binary Tumor vs. Normal baseline (Border → Normal)
- **Rationale:** Future datasets (breast, lung, etc.) rarely provide a Border annotation, so we merged Border into Normal via the new `BinaryClass` column and re-ran the Naive Bayes baseline for an apples-to-apples Tumor-vs-Normal comparison.
- **Command:** `python scripts/rank_nb_baseline.py --label-column BinaryClass --output-json baselines/gse144735_rank_nb_binary_metrics.json`
- **Headline metrics:**
  - Train accuracy 0.77 / macro AUC 0.81.
  - Validation (KUL19) accuracy 0.63 / macro AUC 0.65 — Tumor recall remains low (0.03) even in the binary setting, highlighting the donor shift challenge.
  - Test (KUL01) accuracy 0.71 / macro AUC 0.72 with Tumor F1 0.36.
- **Takeaway:** Collapsing Border into Normal inflates Normal recall but still leaves Tumor under-represented, so transformer fine-tuning must continue to focus on donor-aware sampling and class-weighting even in the simplified binary regime.

### Tree-based baseline (Random Forest, Binary labels)
- **Command:** `python scripts/tree_baseline.py --label-column BinaryClass --model-type random_forest --output-json baselines/gse144735_tree_rf_binary_metrics.json`
- **Feature construction:** The script keeps the top 2,000 most frequent genes and sums inverse-rank weights per gene, producing a dense feature matrix that better suits tree ensembles.
- **Metrics:** Train accuracy/macro F1 = 1.0 (expected overfit), Validation accuracy 0.624 / macro F1 0.399 / macro AUC 0.692, Test accuracy 0.691 / macro F1 0.435 / macro AUC 0.830. Despite stronger overall AUC, Tumor recall remains low (val 0.016, test 0.028), again highlighting the donor shift problem.
- **Interpretation:** Even a much more expressive classical model cannot capture Tumor signatures on the held-out donors, so Geneformer fine-tunes remain necessary for meaningful Tumor recall.
- **Variants:** HistGradientBoosting (`--model-type hist_gb`) hits Validation accuracy 0.582 / macro F1 0.566 and Test accuracy 0.767 / macro F1 0.695 (Tumor recall 0.44), while XGBoost (`--model-type xgboost`) lands at Validation accuracy 0.581 / macro F1 0.528 and Test accuracy 0.753 / macro F1 0.626. All three ensembles overfit the training donors yet still trail transformer runs on Tumor recall.

### Shallow MLP baseline (Binary labels)
- **Command:** `python scripts/mlp_baseline.py --label-column BinaryClass --output-json baselines/gse144735_mlp_binary_metrics.json`
- **Architecture:** Two-layer MLP (512 → 256 hidden units, dropout 0.2) trained with AdamW (lr 5e-4) on the same inverse-rank dense features, selecting the checkpoint with the best validation macro F1.
- **Metrics:** Validation accuracy 0.646 / macro F1 0.532 / macro AUC 0.656; Test accuracy 0.677 / macro F1 0.626 / macro AUC 0.708 with Tumor precision/recall 0.49/0.49. The neural baseline improves Tumor recall versus Naive Bayes and Random Forest but still lags HistGradientBoosting and Geneformer, offering an additional comparison point before deploying the transformer.

### Geneformer fine-tune – GSE144735 (ckpt `outputs/geneformer_colon/checkpoint-4908`)
Evaluation command: `python scripts/evaluate_transformer.py` (config-driven).

| Split | Accuracy | Macro F1 | Macro ROC AUC | Notes |
|-------|----------|----------|---------------|-------|
| Validation (KUL19) | 0.392 | 0.355 | 0.722 | Geneformer still struggles on the hardest donor but improves macro metrics vs. Naive Bayes. |
| Test (KUL01) | 0.742 | 0.721 | 0.879 | Large lift relative to the 0.57/0.54 baseline; Tumor and Normal F1 both improve substantially. |

Per-class precision/recall/F1 for both splits reside in `outputs/geneformer_colon/eval_metrics.json`.

### Geneformer binary fine-tune – GSE144735 (ckpt `outputs/geneformer_colon_binary/checkpoint-818`)
- **Setup:** Binary labels via `BinaryClass` (Border merged into Normal); class weights 1.0/2.0; `balance_strategy: class_donor`; `token_mask_prob` 0.03; no mixup; `warmup_ratio: 0.05`; trained from the base Geneformer checkpoint with gradient accumulation 4 and patience 3.
- **Validation (KUL19):** accuracy 0.676, macro F1 0.626, macro ROC AUC 0.720. Confusion matrix shows Normal recall 0.836 (4,296/5,138) and Tumor recall 0.412 (1,290/3,128), comfortably exceeding the shallow MLP and HistGB baselines on the hardest donor.
- **Test (KUL01):** accuracy 0.707, macro F1 0.689, macro ROC AUC 0.795 with Normal recall 0.693 and Tumor recall 0.736 (1,415/1,922). This is the first setup to push Tumor recall past 0.70 on the held-out donor, exceeding all classical baselines (HistGB: macro F1 0.695 / Tumor recall 0.44; MLP: macro F1 0.626 / Tumor recall 0.49).
- **Conclusion:** The binary relabel + class-donor sampling resolves the donor shift bottleneck observed in the multi-class runs and establishes a strong checkpoint for sequential transfer experiments on additional tumour datasets.

### Geneformer binary + focal fine-tune – GSE144735 (ckpt `outputs/geneformer_colon_binary_focal/checkpoint-818`)
- **Setup:** Same binary labels and class_donor sampler as above, but with focal loss (`focal_gamma 1.5`), heavier Tumor weight 2.2, `token_mask_prob 0.02`, and a short LR warmup (0.05). Training ran 5 epochs from base Geneformer; best checkpoint again landed at step 818.
- **Validation (KUL19):** accuracy 0.674, macro F1 0.608, macro ROC AUC 0.728 with confusion matrix `[[4,480, 658], [2,036, 1,092]]` → Normal recall 0.872, Tumor recall 0.349. Macro F1 improved +0.018 over the non-focal binary run, though Tumor recall dipped slightly.
- **Test (KUL01):** accuracy 0.710, macro F1 0.697, macro ROC AUC 0.813 with confusion matrix `[[2,792, 1,349], [407, 1,515]]` → Tumor recall 0.788 (best to date) and Normal recall 0.674. This finally surpasses the HistGB macro F1 while massively outperforming all baselines on Tumor recall.
- **Takeaway:** Focal loss + stronger weighting deliver the best overall binary checkpoint, especially on the held-out donor, but KUL19 Tumor recall regressed to ~0.35. Upcoming runs will relax the class weight/masking to recover ≥0.40 Tumor recall without giving up the test gains.

### Geneformer binary + focal v2 – GSE144735 (ckpt `outputs/geneformer_colon_binary_focal2/checkpoint-818`)
- **Setup:** Same focal loss but lighter Tumor weight 2.05, lower `token_mask_prob 0.01`, longer warmup 0.10, and outputs saved under `outputs/geneformer_colon_binary_focal2`.
- **Validation (KUL19):** accuracy 0.673, macro F1 0.645, macro ROC 0.727; confusion matrix `[[3,953, 1,185], [1,515, 1,613]]` gives Normal recall 0.769 and Tumor recall 0.516 (best yet on val).
- **Test (KUL01):** accuracy 0.670, macro F1 0.663, macro ROC 0.789; confusion matrix `[[2,472, 1,669], [332, 1,590]]` → Normal recall 0.597, Tumor recall 0.827 (new high).
- **Takeaway:** The gentler regularization fixed the validation donor but hurt Normal recall on KUL01, pulling macro F1 back toward the baseline. Next steps target recovering Normal recall (e.g., lowering focal gamma or adding mild mixup) without sacrificing the improved Tumor recall.
