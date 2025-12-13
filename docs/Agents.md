# Agent Briefing

## Project Snapshot
- Goal: fine-tune a single-cell foundation transformer (Geneformer/scGPT) to classify tumor vs. normal cells using the colorectal cancer scRNA-seq dataset GSE144735, then extend to richer downstream analyses that emphasise biological discovery (pathways, gene programs) rather than pure ML benchmarking.
- Treat the project like a research study that will culminate in a manuscript-quality write-up (methods, experiments, biological interpretation) for the Stanford biomedical data science portfolio. Every step must be logged and reproducible.
- Current assets:
  - `notebooks/01_quality_control.ipynb`: pulls the GEO supplements with `pooch`, constructs an AnnData object, applies baseline QC (cells >=200 genes, genes in >=3 cells), normalizes/log-transforms, flags 5k HVGs, and saves `gse144735_filtered_raw.h5ad` + `gse144735_hvg5k.h5ad`.
  - `README.md`: quickstart instructions, dependency install command, and workflow overview.
    - `scripts/finetune_transformer.py`: Hugging Face Trainer wrapper that ingests the ranked tokens/splits, optionally remaps genes, and fine-tunes a sequence-classification head.
    - `scripts/tree_baseline.py`: Random Forest / HistGradientBoosting / XGBoost baseline that converts ranked genes into dense inverse-rank features for stronger classical references.
    - `scripts/mlp_baseline.py`: shallow PyTorch MLP baseline trained on the same dense features to bridge classical ML and transformer results.
  - `configs/finetune.yaml`: default config consumed by the fine-tuning script so command lines stay short.
  - `.gitignore`: hides `project.md`, `.venv/`, cache directories, and the large data/output folders.

## Environment Expectations
- Preferred local setup: dedicated env (`python -m venv .venv` or `conda create -n geneformer python=3.10`). Install with `pip install -r requirements.txt` (covers Scanpy/pandas/PyTorch; Geneformer is still `pip install -e .` from its repo).
- Register the env as a Jupyter kernel (`python -m ipykernel install --user --name geneformer`) or select the `.venv` interpreter directly in notebooks.
- Colab fallback: run the dependency cell, switch to High-RAM/GPU runtimes. Assume Colab Pro+ access for every session so High-RAM is available.

## Outstanding Work
1. Maintain the run log in `docs/runs.md`. After every fine-tune, append the run name/checkpoint, config tweaks (class weights, sampler, augmentations, etc.), headline metrics (val/test accuracy + macro F1), and the next experiments. This is the lab notebook for the eventual paper.
2. Enhance QC thresholds as needed (mitochondria percentage, doublet filters, upper bounds) and capture rationale in the notebooks/docs.
3. For each new dataset follow the sequence: (i) run `scripts/rank_nb_baseline.py` to log donor-wise baselines; (ii) fine-tune Geneformer from the original HF checkpoint and compare; (iii) zero-shot eval the CRC hub checkpoint on the new test split; (iv) fine-tune starting from the CRC hub and compare against the base run to measure transfer.
4. Compare every fine-tune—whether starting from HF weights or a previously fine-tuned checkpoint—against that dataset’s baseline before changing tokenization or splits.
5. Produce final documentation: detailed methods, experiment log, biological interpretation, and cross-dataset comparisons, culminating in a manuscript-ready narrative for Stanford/CMU.
6. CRC is now binary (Border collapsed into Normal); keep class_donor balancing, focal/mixup tweaks light, and log every change in `docs/runs.md`.
7. Long-range plan: use a hub-and-spoke transfer design (CRC hub -> breast, CRC hub -> lung) instead of long sequential chains; accompany each spoke with zero-shot + base-start controls, then run interpretability to contrast pan-cancer vs tissue-specific programs.
8. Biological discovery log: keep docs/biology_insights.md updated whenever we uncover donor-specific signatures, pan-cancer markers, or planned interpretability analyses so the scientific narrative evolves alongside the engineering log.

## Reminders for Future Sessions
- Check that `pooch` and other deps install in the active kernel; re-run installs + restart runtime if `ModuleNotFoundError` appears.
- Maintain the personal-project tone in notebooks/docs (aligns with the portfolio narrative).
- Keep `project.md` private (ignored by git).
- Before coding, verify repo state via `git status -sb`; avoid touching unrelated user changes.
- Even though lung is larger than CRC, starting from the CRC hub can still help via better inductive bias and faster convergence on shared oncogenic signals. Base-start vs. CRC-start on lung gives a clean comparison. More CRC data would strengthen the hub, but the current checkpoint suffices to measure transfer gains.

## Summary Writing Conventions
- When summarizing completed work (QC, tokenization, training), use journal-style prose (Nature Methods/Bioinformatics tone).
- Structure summaries with concise headers: Dataset and Setting; Data Acquisition; AnnData Assembly; Quality Control; Normalization and Highly Variable Genes; Outputs and Sanity Checks.
- Prefer plain text for method names/paths; reserve inline code formatting for commands only when essential.
- Use US spelling (normalization, tumor) and neutral phrasing ("enabling donor-level cross-validation").
- Report key quantitative results (cell/gene counts after filters, HVG size, basic QC stats) and keep sentences concise.
