# Training History

We keep a running log of every fine-tuning session so future experiments build on what has already been tested.

## 2025-11-07 - Run `geneformer_colon` (checkpoint-1636)
- **Config highlights:** pretrained `ctheodoris/Geneformer`, default class weights (1/1/1), no mixup or extra regularization.
- **Result:** Validation (donor KUL19) accuracy 0.392, macro F1 0.355. Test (donor KUL01) accuracy 0.742, macro F1 0.721.
- **Takeaways:** Model generalizes well to KUL01 but struggles on KUL19 Tumor and Border cells. Logged as the initial benchmark against the Naive Bayes baseline.
- **Next ideas (recorded then):** introduce mild class reweighting, token masking, and mixup to stabilize minority classes.

## 2025-11-10 - Run `geneformer_colon3` (checkpoint-409, best of 4 epochs)
- **Config highlights:** class weights Normal/Border/Tumor = 1.0/1.3/2.0, `token_mask_prob` 0.05, `mixup_prob` 0.3 (`mixup_alpha` 0.6), gradient accumulation 4, patience 3.
- **Result:** Validation accuracy 0.408, macro F1 0.402; Test accuracy 0.757, macro F1 0.756. Tumor recall improved markedly on test but KUL19 Border/Tumor recall remains low.
- **Takeaways:** Mixup plus modest weighting helped overall metrics but did not fix the donor shift. The KUL19 confusion matrix shows Tumor predictions still collapsing into Normal/Border.
- **Next ideas (recorded now):** explore stronger class weighting, balanced sampling, or focal loss to emphasize Tumor/Border without duplicating cells.

## Planned Next Experiment - Balanced class sampling plus stronger weights
- Enable class-balanced sampling so each training batch draws Tumor and Border cells more frequently while still mixing donors.
- Increase Tumor/Border loss weights (1.0/1.6/2.5) to emphasize minority classes.
- Keep focal loss disabled for now (`focal_gamma` 0.0) but the code now supports enabling it if we need sharper focus on hard examples.
- Goal: lift KUL19 Tumor recall without sacrificing KUL01 performance or resorting to donor-level oversampling.

## 2025-11-11 - Run `geneformer_colon4` (balanced sampling, paused after epoch 1)
- **Config highlights:** starting from `outputs/geneformer_colon3/checkpoint-409`, outputting to `outputs/geneformer_colon4`; class weights 1.0/1.6/2.5; `balance_strategy: class`; `token_mask_prob` 0.05; `mixup_prob` 0.3; `focal_gamma` 0.0.
- **Status:** training underway. After epoch 1 the validation metrics dipped (accuracy 0.363, macro F1 0.314) as the sampler rebalanced batches. Session paused for the night; resume from the same checkpoint next run to observe whether metrics recover on epoch 2+.
- **Action items:** if macro F1 stays below the previous run after epoch 2, consider (a) reducing class weights slightly, (b) lowering mixup probability, or (c) switching to `balance_strategy: class_donor`. Log final metrics and analysis here once the run finishes.

## 2025-11-12 - Run `geneformer_colon5` (class-donor balancing, best checkpoint epoch 1)
- **Config highlights:** `outputs/geneformer_colon3/checkpoint-409` ➜ `outputs/geneformer_colon5`; `balance_strategy: class_donor`; class weights 1.0/1.4/2.2; `token_mask_prob` 0.03; `mixup_prob` 0.2; `focal_gamma` 0.0; patience 3 (max 8 epochs).
- **Epoch-by-epoch:**  
  - Epoch 1: val acc 0.416 / macro F1 0.405 / macro ROC 0.734 (`eval_loss` 1.97).  
  - Epoch 2: val acc 0.383 / macro F1 0.363 / macro ROC 0.748 (`eval_loss` 2.56).  
  - Epoch 4: val acc 0.360 / macro F1 0.326 / macro ROC 0.696 (`eval_loss` 3.36). Training loss kept falling (0.25) but validation degraded, so early stopping kept the epoch-1 checkpoint (`outputs/geneformer_colon5/checkpoint-409`) as best.
- **Final metrics (best ckpt @ epoch 1):** Validation 0.416 acc / 0.405 macro F1 / 0.734 macro ROC. Test 0.728 acc / 0.724 macro F1 / 0.876 macro ROC; Tumor precision/recall improved slightly vs. run3 while Border dipped (confusion matrices pending full evaluation dump).
- **Interpretation:** Donor-aware sampling plus lighter weights avoided the dramatic collapse seen in run4, but validation never surpassed run3’s 0.408/0.402. Test macro F1 (0.724) trails run3 by ~0.03, indicating that mixup 0.2 + class_donor sampling might still be too strong. Training loss suggests the model keeps fitting but the val donor remains a bottleneck.
- **Next steps:** Try (a) lowering `mixup_prob` to 0.15 and/or (b) enabling focal loss (`focal_gamma` 1.5) to focus on hard Tumor examples without further boosting sampler weights. If that still plateaus, consider reinitialising the classifier head or restarting from the original Geneformer checkpoint for a clean slate before stacking more regularization.

## Planned Next Experiment - `geneformer_colon6` (no mixup baseline)
- Output directory: `outputs/geneformer_colon6` starting from the original `ctheodoris/Geneformer` weights so we can measure a clean run without inherited biases from colon3.
- Keep `balance_strategy: class_donor`, class weights 1.0/1.4/2.2, and `token_mask_prob` 0.03, but set `mixup_prob` to 0 so every batch uses real cells only.
- Success criteria: validation macro F1 ≥ 0.40 by epochs 2–3 and stable thereafter, and/or test macro F1 ≥ 0.72. If those hit, we can revisit lighter mixup or focal loss; if not, we will evaluate classifier-head reinit or different sampling.
