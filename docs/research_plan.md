# Research Plan: Single-Cell Tumor Classification

## Objectives
1. Establish a reproducible end-to-end pipeline (QC + tokenization + splitting + baselines + transformer fine-tune) on GSE144735 to demonstrate donor-level tumor vs. normal classification, while surfacing biological discoveries (gene programs, donor shifts, pathway signatures) rather than just benchmarking ML metrics.
2. Iteratively improve Geneformer fine-tuning with principled experiments (class weighting, balanced sampling, focal loss, mild mixup) and log every run in `docs/runs.md`.
3. Evaluate the final colorectal checkpoint against the Naive Bayes baseline and earlier Geneformer runs; capture confusion matrices and per-class metrics for manuscript tables.
4. Pivot to pairwise transfer (hub-and-spoke): treat the strongest CRC checkpoint as the hub and measure transfer to new cancers (breast, lung) via zero-shot evaluation and base-vs-CRC-start fine-tunes to quantify biological distance.
5. Synthesize the above into a paper-style report for the Stanford/CMU biomedical data science application (Introduction, Methods, Results, Discussion + interpretability).

## Current Status (2025-12-12)
- **CRC phase complete:** QC/tokenization/splits, baselines, and multiple Geneformer checkpoints finished; CRC hub checkpoint will seed transfer to other tissues.
- **Lung (GSE131907) in progress:** QC/tokenization/splits completed (80/10/10 patients: 46/5/7). Tokens under `gse131907/processed/tokens/`; splits logged in `splits_by_patient.npz` + `splits_summary.tsv`.
- **Baselines (CRC):** Rank-based Naive Bayes, tree ensembles, and shallow MLP logged for CRC.
- **Transformer runs (CRC):** Multi-class runs plateaued; binary + focal variants achieved the best Tumor recall on held-out donors (`geneformer_colon_binary`, `_binary_focal`, `_binary_focal2`).
- **Lung baselines:** BinaryClass baselines logged on lung: Naive Bayes (test acc 0.690 / macro AUC 0.753), HistGB (test acc 0.889 / macro F1 0.843 / macro AUC 0.970), XGBoost (test acc 0.875 / macro F1 0.817 / macro AUC 0.964), shallow MLP (test acc 0.826 / macro F1 0.758 / macro AUC 0.906).
- **Transfer rationale:** Even though lung is larger than CRC, starting from the CRC hub can still improve inductive bias and convergence on shared oncogenic signals. Base-start vs. CRC-start on lung is the clean comparison; more CRC data would strengthen the hub, but the current checkpoint suffices to measure transfer gains.
- **Lung transfer experiments:** Zero-shot CRC hub (focal2) on lung: val acc 0.488 / macro F1 0.433 / macro AUC 0.376; test acc 0.696 / macro F1 0.659 / macro AUC 0.728 (remap matched ~16.9k genes; ~38.5% missing). Fine-tunes next: run base-start on lung using CRC focal2-style hyperparams adjusted for lung balance (Normal 1.2 / Tumor 1.0, focal_gamma 1.5, token_mask_prob 0.01, mixup off, class_donor, lr 2e-5, warmup 0.1), then CRC-start with the focal2 checkpoint once a full checkpoint is available.

## Upcoming Experiments
1. Lung base-start fine-tune (Geneformer base) with focal2-style hyperparams adjusted for lung class balance.
2. Lung CRC-start fine-tune (once a full focal2 checkpoint is available) with the same hyperparams to measure transfer gain over base-start.
3. Compare both lung runs against lung baselines and zero-shot results; log metrics and confusion matrices.
4. Proceed to breast dataset with the same protocol (zero-shot, base-start, CRC-start).

## Documentation Checklist
- [x] QC/tokenization/splits docs for CRC.
- [x] Lung tokenization and splits complete; baselines logged.
- [ ] Lung fine-tune runs logged (base-start and CRC-start).
- [ ] Transfer comparison tables/figures drafted.
