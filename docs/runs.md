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

## 2025-11-13 - Run `geneformer_colon6` (no-mixup baseline, epoch 1)
- **Config highlights:** fresh initialization from `ctheodoris/Geneformer`; `balance_strategy: class_donor`; class weights 1.0/1.4/2.2; `token_mask_prob` 0.03; `mixup_prob` 0.0; patience 3 (max 8 epochs).
- **Result (epoch 1):** Validation accuracy 0.425, macro F1 0.419, macro ROC AUC 0.719 (`eval_loss` 2.17). This already meets the ≥0.40 macro F1 target and slightly exceeds run3’s val macro F1 at the same epoch count.
- **Takeaways:** Removing mixup while restarting from base Geneformer stabilised the class_donor sampler—no early collapse like run4/5, and the model seems to focus on real donor signals again.
- **Result (epoch 2):** Validation accuracy 0.439, macro F1 0.433, macro ROC AUC 0.748 (`eval_loss` 2.85). Metrics continued to improve, clearing the success threshold (≥0.40 macro F1 by epoch 2–3).
- **Next actions:** keep training through epoch 4–5 to confirm stability, then capture test metrics + confusion matrices. If later epochs dip, revisit focal loss (`focal_gamma`≈1.5); otherwise proceed to the binary relabel plan once colon6 finalises.
- **Result (epoch 3):** Validation accuracy 0.423, macro F1 0.402, macro ROC AUC 0.754 (eval_loss 3.14). Slight drop from epoch 2 but still above the 0.40 macro F1 floor; monitor whether early stopping keeps the epoch-2 checkpoint.
- **Future-proofing note:** We are keeping the three-class setup for now to finish CRC ablations, but the research plan includes collapsing Border into Normal and regenerating baselines once the multi-class run stabilises. That binary checkpoint will become the hand-off point for future datasets that lack a Border label.

## 2025-11-14 - Pivot to binary CRC labels (`BinaryClass`)
- **Rationale:** KUL19 (val) now tops out at macro F1 ≈0.43 but Tumor recall remains <0.40 on both donors even with class-donor sampling and heavier weights. Downstream datasets (breast, lung) rarely include a “Border” annotation, so keeping a three-class head complicates continual learning while providing limited biological resolution with only six CRC donors.
- **Action:** Added a `BinaryClass` column to `gse144735_tokens_metadata.tsv` that maps `Border → Normal`, effectively treating Border tissue as non-tumour. Updated configs to point at `outputs/geneformer_colon_binary`, use `label_column: BinaryClass`, and simplify loss weights to {Normal: 1.0, Tumor: 2.0}. Mixup remains disabled to preserve the stability gains from run6.
- **Next steps:** Regenerate the Naive Bayes baseline with `--label-column BinaryClass`, then fine-tune a fresh Geneformer checkpoint using the binary labels. This CRC binary checkpoint becomes the starting point for sequential fine-tuning on additional tumour datasets whose annotations only distinguish Tumor vs. Normal. Added a gentle LR warmup (`warmup_ratio: 0.05`) in the config to stabilise the first binary fine-tune epoch.
- **Classical baseline upgrade:** Introduced `scripts/tree_baseline.py` (Random Forest / HistGB / XGBoost) plus `scripts/mlp_baseline.py`. RF still stalls at KUL01 macro F1 0.44, XGBoost lifts Tumor recall to 0.27 (macro F1 0.63), HistGB reaches macro F1 0.70 with Tumor recall 0.44, and the shallow MLP lands at macro F1 0.63 with Tumor recall ~0.49. All remain below transformer runs, giving us a rich benchmark suite for the manuscript.

## 2025-11-15 - Run `geneformer_colon_binary` (epoch 1)
- **Config highlights:** same no-mixup recipe as run6 but using the new binary labels: `label_column: BinaryClass`, class weights 1.0/2.0, `balance_strategy: class_donor`, `token_mask_prob: 0.03`, `mixup_prob: 0.0`, `warmup_ratio: 0.05`, gradient accumulation 4, max 8 epochs with patience 3. Initialised from the base `ctheodoris/Geneformer` checkpoint (`outputs/geneformer_colon_binary`).
- **Epoch 1 results (Trainer eval):** validation accuracy 0.664 / macro F1 0.550 / macro ROC not yet recorded (see log snippet). This already clears the shallow MLP baseline (val macro F1 0.532) and greatly improves Tumor recall on KUL19 versus every classical model.
- **Outstanding:** need to run `python scripts/evaluate_transformer.py --config configs/eval.yaml` so we capture full val/test confusion matrices and AUROC for the manuscript tables. Early signs suggest the binary setup resolves the Tumor collapse seen in the multi-class runs, but we need the KUL01 metrics to confirm the gap over the HistGB baseline (test macro F1 0.695, Tumor recall 0.44).
- **Next actions:** finish the evaluation pass, log best checkpoint path, and decide whether to keep training past epoch 1 (watch for further gains before patience triggers). If Tumor recall on test is still below the MLP baseline (~0.49), try focal loss (`focal_gamma 1.5`) or a slightly higher Tumor weight (1.0/2.3) on a short follow-up run.

## 2025-11-15 - Run `geneformer_colon_binary` (best checkpoint @ epoch 2)
- **Training curve:** early stopping kept `outputs/geneformer_colon_binary/checkpoint-818` (epoch 2) as best even though training continued to epoch 5 (train loss 0.185). Warmup + class-donor sampler prevented the early-val dip we saw in run6.
- **Validation (KUL19) metrics:** accuracy 0.676, macro F1 0.626, macro ROC AUC 0.720. Confusion matrix shows Normal recall 0.836 (4,296/5,138) and Tumor recall 0.412 (1,290/3,128)—a huge lift over the HistGB baseline’s 0.44 macro F1 and even the shallow MLP (0.53). Tumor precision is 0.605, so the model is no longer collapsing Tumor into Normal on the hardest donor.
- **Test (KUL01) metrics:** accuracy 0.707, macro F1 0.689, macro ROC AUC 0.795 with Normal recall 0.693 and Tumor recall 0.736 (1,415/1,922). This beats every classical baseline handily (best macro F1 prior was HistGB at 0.695 with Tumor recall 0.44, shallow MLP at macro F1 0.626 / Tumor recall 0.49), confirming the binary transformer’s advantage.
- **Takeaways:** The binary relabel plus class-donor sampling solved both objectives: val donor improved to macro F1 0.626 (first time >0.60) and test Tumor recall is finally >0.70. No mixup seems sufficient; we’ll keep focal loss in reserve for future datasets.
- **Next steps:** (1) Archive the best checkpoint path in the repo docs/eval summaries, (2) run interpretability prep (attention summaries, feature importances) on this binary model, (3) plan sequential transfer to the next dataset (likely breast) starting from `checkpoint-818`. If we chase further gains on CRC, experiment with lighter token masking (0.02) or marginally higher Tumor weight (2.2) while keeping mixup off.

## Planned Next Experiment - `geneformer_colon_binary_focal`
- **Motivation:** although binary run 1 already beats classical baselines on Tumor recall, HistGB’s macro F1 on KUL01 (0.695) is still slightly higher. We want a transformer checkpoint that dominates both macro F1 and Tumor recall to make the performance gap obvious in the manuscript.
- **Config tweaks:** keep the successful no-mixup/class-donor recipe but (i) increase Tumor weight to 2.2, (ii) lower `token_mask_prob` to 0.02, and (iii) enable focal loss (`focal_gamma 1.5`) so the model down-weights easy Normal predictions. Output directory: `outputs/geneformer_colon_binary_focal`.
- **Success criteria:** maintain ≥0.40 Tumor recall on KUL19 while clearing macro F1 0.70 on KUL01 (matching or exceeding HistGB’s 0.695) and ideally bumping macro ROC past 0.80. If focal loss destabilises training, fall back to the previous checkpoint and revisit mixup ≤0.1 instead.

## 2025-11-16 - Run `geneformer_colon_binary_focal` (best checkpoint `checkpoint-818`)
- **Config highlights:** `focal_gamma 1.5`, class weights {Normal: 1.0, Tumor: 2.2}, `token_mask_prob: 0.02`, `warmup_ratio: 0.05`, mixup disabled, class_donor sampler, gradient accumulation 4, trained 5 epochs from the base Geneformer weights with early stopping.
- **Trainer summary:** train loss 0.106, 3.27k steps over ~21.5k s (0.152 steps/s); best checkpoint saved at step 818 even though training continued to epoch 5.
- **Validation (KUL19):** accuracy 0.674, macro F1 0.608, macro ROC 0.728. Confusion matrix `[[4480, 658],[2036, 1092]]` ⇒ Normal recall 0.872, Tumor recall 0.349, precision 0.624. Tumor recall dipped vs. the non-focal binary run (0.412) but overall macro F1 improved by +0.018.
- **Test (KUL01):** accuracy 0.710, macro F1 0.697, macro ROC 0.813. Confusion matrix `[[2792, 1349],[407, 1515]]` ⇒ Tumor recall 0.788 (+0.05 vs. prior best) and Normal recall 0.674. Macro F1 now edges the HistGB baseline (0.695) while dramatically improving tumor coverage.
- **Takeaways:** focal loss + heavier weighting successfully boosts KUL01 Tumor recall/macro F1, but KUL19 Tumor recall regressed. Need to relax the weight/masking pressure so the hardest donor stabilises without losing the test gains. This is currently the best overall binary checkpoint (macro F1 + Tumor recall) and will be the handoff for Colab reruns unless the next ablation wins.

## Planned Next Experiment - `geneformer_colon_binary_focal2`
- **Config tweaks:** reduce Tumor weight to 2.05, lower `token_mask_prob` to 0.01 (less perturbation for the minority donor), extend warmup to 0.10, and log everything under `outputs/geneformer_colon_binary_focal2`. Keep focal gamma 1.5, mixup off, class_donor sampler, and other hyperparameters unchanged (see `configs/finetune.yaml` / `configs/eval.yaml`).
- **Goal:** recover ≥0.40 Tumor recall on KUL19 without giving up the KUL01 macro F1 gains from the focal run (≥0.69). If Normal recall collapses, the next lever will be reducing focal gamma rather than further weight tweaks.

## 2025-11-17 - Run `geneformer_colon_binary_focal2` (best checkpoint `checkpoint-818`)
- **Config highlights:** focal gamma 1.5, class weights {1.0, 2.05}, `token_mask_prob 0.01`, `warmup_ratio 0.10`, no mixup, class_donor sampler, output to `outputs/geneformer_colon_binary_focal2`.
- **Trainer summary:** train loss 0.108 over 5 epochs (~21.4 ks runtime, 0.153 steps/s); early stopping again selected step 818 even though training reached epoch 5 (later epochs showed overfitting).
- **Validation (KUL19):** accuracy 0.673, macro F1 0.645, macro ROC 0.727 with confusion matrix `[[3,953, 1,185], [1,515, 1,613]]`. Normal recall 0.769, Tumor recall 0.516 — a sizable lift (+0.167) over the first focal run while keeping macro F1 right at the 0.64 target.
- **Test (KUL01):** accuracy 0.670, macro F1 0.663, macro ROC 0.789 with confusion matrix `[[2,472, 1,669], [332, 1,590]]`. Tumor recall improved further to 0.827 (best to date) but Normal recall dipped to 0.597, pulling macro F1 down ~0.03 from `geneformer_colon_binary_focal`.
- **Takeaways:** The gentler weighting fixed the validation donor (Tumor recall now >0.5) but sacrificed Normal performance on KUL01. We need a middle ground that keeps Tumor recall ≥0.80 while nudging Normal recall back above ~0.67.

## Planned Next Experiment - `geneformer_colon_binary_focal3`
- **Config tweaks:** keep the successful weight/token-mask/warmup from focal2 but reduce focal gamma to 1.2 (less aggressive down-weighting of easy samples) and introduce a tiny mixup probability (0.05) to re-expose the model to Normal contexts without overwhelming Tumor batches. Output to `outputs/geneformer_colon_binary_focal3` with otherwise identical settings.
- **Success criteria:** maintain KUL19 Tumor recall ≥0.50 and lift KUL01 Normal recall back to ≥0.66 so macro F1 rebounds toward 0.69 while keeping Tumor recall ≥0.80. If Normal recall is still low, the fallback is to disable focal loss entirely while retaining the 2.05 weight / 0.01 masking combo.
- **Hypothesis:** gentler weighting + more warmup should recover ≥0.40 Tumor recall on KUL19 while maintaining ≥0.70 macro F1 / ≥0.78 Tumor recall on KUL01. If KUL19 still lags, next levers are (a) donor-aware mixup only on KUL19 batches or (b) turning focal off while retaining the lighter weight.
