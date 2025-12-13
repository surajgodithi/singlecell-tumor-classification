# Research Plan: Single-Cell Tumor Classification

## Objectives
1. Establish a reproducible end-to-end pipeline (QC + tokenization + splitting + baselines + transformer fine-tune) on GSE144735 to demonstrate donor-level tumor vs. normal classification, while surfacing biological discoveries (gene programs, donor shifts, pathway signatures) rather than just benchmarking ML metrics.
2. Iteratively improve Geneformer fine-tuning with principled experiments (class weighting, balanced sampling, focal loss, mild mixup) and log every run in `docs/runs.md`.
3. Evaluate the final colorectal checkpoint against the Naive Bayes baseline and earlier Geneformer runs; capture confusion matrices and per-class metrics for manuscript tables.
4. Pivot to pairwise transfer (hub-and-spoke): treat the strongest CRC checkpoint as the hub and measure transfer to new cancers (breast, lung) via zero-shot evaluation and base-vs-CRC-start fine-tunes to quantify biological distance.
5. Synthesize the above into a paper-style report for the Stanford/CMU biomedical data science application (Introduction, Methods, Results, Discussion + interpretability).

## Current Status (2025-11-22)
- **CRC phase complete:** QC/tokenization/splits, baselines, and multiple Geneformer checkpoints finished; CRC hub checkpoint will seed transfer to other tissues.
- **Lung (GSE131907) in progress:** QC notebook executed in Colab with chunked loading and label standardization (`Class = Sample_Origin`, `Patient = Sample`); awaiting upload of `gse131907/processed/gse131907_filtered_raw.h5ad` and `gse131907_hvg5k.h5ad` to proceed locally with tokenization/splits.
- **Baselines (CRC):** Rank-based Naive Bayes (`baselines/gse144735_rank_nb*.json`), tree ensembles, and shallow MLP all logged for CRC.
- **Transformer runs (CRC):** Multi-class runs plateaued; binary + focal variants achieved the best Tumor recall on held-out donors (`geneformer_colon_binary`, `_binary_focal`, `_binary_focal2`).
- **Documentation:** `docs/runs.md` tracks CRC history and the lung handoff; `docs/Agents.md` maintains project framing.
- **Binary relabel groundwork:** `BinaryClass` (Border→Normal) wired through scripts/configs for reuse on new datasets.
- **Lung baselines:** Completed donor-wise splits (train 46 / val 5 / test 7 patients; 80/10/10). BinaryClass baselines logged: Naive Bayes (test acc 0.690 / macro AUC 0.753), HistGB (test acc 0.889 / macro F1 0.843 / macro AUC 0.970), XGBoost (test acc 0.875 / macro F1 0.817 / macro AUC 0.964), and shallow MLP (test acc 0.826 / macro F1 0.758 / macro AUC 0.906).
- **Transfer rationale:** Even though lung is larger than CRC, starting from the CRC hub can still improve inductive bias and convergence on shared oncogenic signals. Base-start vs. CRC-start on lung is the clean comparison; more CRC data would strengthen the hub, but the current checkpoint suffices to measure transfer gains.

## Upcoming Experiments
1. Fix the hub choice: use `geneformer_colon_binary_focal2` (Tumor-strong) as the default CRC hub; keep `geneformer_colon_binary_focal` as a balanced alternate. Archive focal3 as a Normal-leaning variant.
2. Launch the interpretability sprint on the CRC hub (attention summaries, saliency/rank importances, donor-specific signatures) so manuscript figures start taking shape.
3. Pairwise transfer roadmap: for breast and lung datasets, run zero-shot CRC hub evaluation, base->target fine-tune, and CRC->target fine-tune; compare convergence and final metrics to quantify biological proximity.
4. Keep baselines honest: replicate Naive Bayes / tree / MLP baselines on each new dataset so transformer gains remain contextualised.
5. Interpretability sprint (multi-tissue): after pairwise transfer, aggregate attention/gene-importance scores and UMAP embeddings across tissues to contrast pan-cancer vs. tissue-specific programs and discuss any forgetting vs. positive transfer.

## Paper Outline (Working Draft)
1. **Introduction:** motivation for cross-cancer tumor classification with foundation models; challenges of donor shift.
2. **Methods:** data acquisition, QC, tokenization, donor splits, baseline classifier, Geneformer fine-tune details (augmentations, balanced sampling, focal loss).
3. **Results:** tables comparing Naive Bayes vs. Geneformer runs (val/test metrics, confusion matrices); discussion of mitigation strategies for KUL19.
4. **Discussion:** lessons on class imbalance, donor shift, and pathways for extending to other cancers (pairwise transfer vs. joint training).
5. **Future Work:** evaluation on breast/lung datasets, domain adaptation, biological interpretation.

## Documentation Checklist
- [x] QC/tokenization/splits docs.
- [x] Baseline metrics stored and referenced.
- [x] Training history started in `docs/runs.md`.
- [ ] Finalize `geneformer_colon4` entry (metrics + analysis).
- [ ] Add evaluation docs for best checkpoints (per-class tables ready for the manuscript).
- [ ] Draft manuscript sections once colorectal experiments stabilize.
