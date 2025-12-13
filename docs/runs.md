GSE144735 First Training run:
--- VAL ---
Accuracy: 0.392, Macro F1: 0.355, Macro ROC AUC: 0.722
Confusion matrix (rows=true, cols=pred):
[[1213 1884  166]
 [  94 1764   17]
 [1943  923  262]]
100% 758/758 [08:33<00:00,  1.48it/s]

--- TEST ---
Accuracy: 0.742, Macro F1: 0.721, Macro ROC AUC: 0.879
Confusion matrix (rows=true, cols=pred):
[[1866   20  243]
[  39 1944   29]
 [1217   15  690]]


## 2025-11-22 - Lung dataset prep (GSE131907)
- QC notebook now loads `GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz` in 2k-row sparse chunks to avoid RAM issues and standardizes labels (`Class = Sample_Origin`, `Patient = Sample`). QC metrics recompute `n_genes_by_counts` when absent.
- Class distribution observed in QC: tLung 45,149; nLung 42,995; nLN 37,446; mBrain 29,060; mLN 21,479; PE 20,304; tL/B 12,073.
- Tokenization notebook maps `Sample_Origin` to `Class` and adds `BinaryClass`, collapsing tumor-like labels (tLung, tL/B, mBrain, mLN, PE) into Tumor and normal-like (nLung, nLN) into Normal; `Sample` is used as a patient proxy pending a donor column.
- Next: rerun tokenization and splits in Colab, then baselines and zero-shot/base-start/CRC-start once splits are saved. Adjust the mapping if we decide to drop metastasis/effusion from the Tumor bin.
- Tokenization now completed locally: `gse131907/processed/tokens/` contains gene vocab, ranked tokens (2048 max genes; lengths median ~1103), and metadata with `BinaryClass` (Tumor 128,065 / Normal 80,441). Per-patient/class counts are intact (e.g., top rows: LUNG_N20/nLung 5,798; NS_07/mBrain 5,730; LN_07/nLN 5,713). Donor-wise splits regenerated locally with an 80/10/10 patient allocation (58 patients → train 46, val 5, test 7; cells train 164,672 / val 17,956 / test 25,878) in `gse131907/processed/tokens/splits_by_patient.npz` + `splits_summary.tsv`.
- Baselines (BinaryClass) on lung (train 46 / val 5 / test 7 patients):
  - Naive Bayes: test acc 0.690 / macro AUC 0.753; Tumor F1 0.808, Normal F1 0.190.
  - HistGradientBoosting (2k top genes, n_estimators=300): test acc 0.889 / macro F1 0.843 / macro AUC 0.970; Tumor F1 0.928, Normal F1 0.757.
  - XGBoost (2k top genes, n_estimators=300): test acc 0.875 / macro F1 0.817 / macro AUC 0.964; Tumor F1 0.920, Normal F1 0.714.
  - Shallow MLP (2k top genes, 512/256, 20 epochs): test acc 0.826 / macro F1 0.758 / macro AUC 0.906; Tumor F1 0.886, Normal F1 0.631.

## 2025-11-22 - Status update
- Colorectal phase complete: CRC QC/tokenization/splits/baselines + Geneformer runs are done and will now serve as the hub checkpoint for transfer.
- Currently pivoted to lung (GSE131907): QC notebook run in Colab; awaiting upload of the processed AnnData artifacts (`gse131907/processed/*.h5ad`) to continue locally.
- Next steps once data lands: run tokenization notebook to produce ranked tokens + metadata, generate donor-wise splits, and compute lung baselines before the hub-and-spoke comparison (CRC hub vs base-start fine-tunes).
- CRC → Lung zero-shot (focal2 hub, BinaryClass, 46/5/7 patients): val acc 0.488 / macro F1 0.433 / macro AUC 0.376 (confusion [[7181, 3156], [6032, 1587]]); test acc 0.696 / macro F1 0.659 / macro AUC 0.728 (confusion [[4746, 2410], [5461, 13261]]). Vocab remap matched ~16.9k genes via alias map; ~38.5% missing. Next: lung fine-tunes (base-start vs CRC-start) with CRC-best hyperparams.
- Lung fine-tunes planned: config now points to lung tokens/output and uses CRC focal2-style settings with weights adjusted for lung’s class balance (Normal 1.2 / Tumor 1.0, focal_gamma 1.5, token_mask_prob 0.01, mixup off, class_donor sampler). Base-start run will use `ctheodoris/Geneformer`; CRC-start will use the CRC focal2 checkpoint once a full checkpoint (with config.json) is available.

3 Training Run:
--- VAL ---
Accuracy: 0.408, Macro F1: 0.402, Macro ROC AUC: 0.656
Confusion matrix (rows=true, cols=pred):
[[1024 1703  536]
 [ 141 1598  136]
 [1621  758  749]]
100% 758/758 [08:30<00:00,  1.48it/s]

--- TEST ---
Accuracy: 0.757, Macro F1: 0.756, Macro ROC AUC: 0.893
Confusion matrix (rows=true, cols=pred):
[[1635   17  477]
 [  43 1883   86]
 [ 842    7 1073]]
