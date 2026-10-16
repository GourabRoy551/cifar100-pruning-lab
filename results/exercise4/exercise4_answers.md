# Exercise 4: Re-training a pruned model

**Code:** `pruning.py`, Exercise 4 block (`run_exercises = [4]`), functions `load_pruned_model()` and `fine_tune()`
**Input models:** the 12 globally pruned models saved in Exercise 3, with their masks (`results/exercise3/exercise3_global_XX.pt`)
**Evaluation data:** validation set (10,000 images), the same saved split and pre-processing as for training

## 1. Task (from the template)

For each model saved in Exercise 3, retrain the network on the training set **for a few epochs with a lower learning rate**. Then draw the accuracy / sparsity curve, comment on it, and choose a model.

## 2. What was done

| Setting | Value |
|---|---|
| Models retrained | All 12 Exercise 3 models (0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90% global sparsity) |
| Training data | The 40,000 training images of the saved split, same augmentation as `train.py` (random crop + flip) |
| Epochs | **5**, against 30 for the baseline |
| Learning rate | **0.01**, 10× lower than the baseline's 0.1, with cosine decay to almost 0 over the 5 epochs |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4 (as for the baseline) |
| Seed | 42, reset before each model, so all models see the same batch order |
| Model kept for each ratio | The model after the **last** epoch, not the best epoch, to avoid choosing an epoch on the same validation set that is then reported |
| Rule for choosing a model | Highest sparsity with validation accuracy at most 1 point below the baseline (≥ 67.67%) |

**How the pruned weights stay at zero:**
- The models are loaded **with their masks**. In every forward pass PyTorch uses `weight = weight_orig × weight_mask`, and the optimizer only updates `weight_orig`.
- A pruned position therefore always gives 0 and receives no gradient.
- **Check:** the sparsity measured after retraining equals the target at every ratio (0.00%, 5.00%, …, 90.00%), so no pruned weight came back.

**Control experiment:** the 0% model (nothing pruned) goes through exactly the same retraining. It shows the effect of the retraining procedure itself, separately from the pruning.

## 3. Results

Baseline (unpruned): **68.67%** validation accuracy, loss 1.1873.

| Sparsity | Before re-training (Ex. 3) | After re-training | Gain (points) | Gap to baseline (points) | Val. loss after | Train acc. (last epoch) |
|---:|---:|---:|---:|---:|---:|---:|
| 0% (control) | 68.67% | 68.41% | −0.26 | 0.26 | 1.2370 | 94.43% |
| 5% | 68.67% | 68.39% | −0.28 | 0.28 | 1.2392 | 94.46% |
| 10% | 68.77% | 68.35% | −0.42 | 0.32 | 1.2375 | 94.42% |
| 15% | 68.79% | 68.33% | −0.46 | 0.34 | 1.2381 | 94.43% |
| 20% | 68.61% | 68.51% | −0.10 | 0.16 | 1.2355 | 94.34% |
| 30% | 68.78% | 68.40% | −0.38 | 0.27 | 1.2340 | 94.22% |
| 40% | 68.21% | 68.66% | +0.45 | 0.01 | 1.2274 | 94.05% |
| 50% | 67.67% | 68.41% | +0.74 | 0.26 | 1.2227 | 93.64% |
| 60% | 65.09% | 68.52% | +3.43 | 0.15 | 1.2155 | 92.90% |
| **70%** | 58.22% | **68.05%** | **+9.83** | **0.62** | 1.2132 | 91.55% |
| 80% | 34.60% | 67.39% | +32.79 | 1.28 | 1.2214 | 88.35% |
| 90% | 4.61% | 65.31% | **+60.70** | 3.36 | 1.2511 | 79.29% |

**Figures:**
- `exercise4_curve.png/.pdf`: accuracy and loss vs sparsity, before (Ex. 3) and after retraining.
- `exercise4_retraining_history.png/.pdf`: validation accuracy at each retraining epoch for 50–90% sparsity.

The full epoch-by-epoch numbers are in `exercise4_history.csv` and `log.txt`.

## 4. Answers

### [QUESTION 1] Draw the accuracy / sparsity ratio curve and comment on the results.

**1. Retraining recovers almost all the accuracy lost by pruning.**
- After retraining, the curve is **almost flat**: between 68.0% and 68.7% from 0% up to **70% sparsity**. It then decreases only slowly, to 67.39% at 80% and 65.31% at 90%.
- Without retraining (Exercise 3), the same models fall to 58%, 35% and 5% at 70, 80 and 90%.
- **The more a model was damaged, the more it gains:**

  | Sparsity | Gain from retraining |
  |---:|---:|
  | 60% | +3.4 points |
  | 70% | +9.8 points |
  | 80% | +32.8 points |
  | 90% | +60.7 points |

  The 90% model goes from almost random (4.61%) to 65.31% in only 5 epochs, with **only 10% of the weights** (282k of 2.82M).

**2. Most of the recovery happens in the first epoch** (see `exercise4_retraining_history.png`):
- **80% model:** 34.6% → 65.1% after one epoch.
- **90% model:** 4.6% → 61.9% after one epoch.
- **Why:** the remaining weights already hold most of the knowledge. The network only needs to readjust them, especially the BatchNorm statistics and scales, to make up for the removed weights. It does not need to relearn from scratch.
- **The remaining epochs** add a few more points while the learning rate decreases.

**3. At low sparsity (0–30%), retraining gives nothing, or even −0.1 to −0.46 points.**
- **The control explains it:** the **unpruned 0% model also loses 0.26 points** with the same retraining.
- **Why the procedure itself costs a little:**
  - The baseline finished its training with a learning rate close to 0 (end of the cosine schedule). Restarting at lr 0.01 moves the model away from its converged state: validation accuracy drops to about 66% after epoch 1, then climbs back to about 68.4% as the learning rate decreases again.
  - Five more epochs on the same training data also increase the **overfitting**: training accuracy rises from 92.4% to 94.4%, and validation loss rises from 1.187 to about 1.24.
- **Compared with this control,** the pruned models up to 60% perform **the same as the retrained unpruned model** (68.3–68.7% against 68.41%). Pruning up to 60% costs nothing once the model is retrained. The differences of ±0.2 points are within noise (0.1 point = 10 images).

**4. Pruning acts a bit like a regulariser at high sparsity.**
- The more weights are removed, the **lower the training accuracy** after retraining (94.4% at 0% → 91.6% at 70% → 79.3% at 90%), while validation accuracy stays similar up to 70%.
- The gap between training and validation accuracy therefore shrinks (about 26 points at 0%, about 23.5 at 70%, about 14 at 90%).
- The validation loss is even slightly lower at 60–80% (1.21–1.22) than for the retrained 0% model (1.24).
- **Interpretation:** a network with fewer weights memorises the training set less. This is a single run, so it is an observation, not a proven effect.

**5. Where the curve bends:** up to 70% the retrained models stay within 1 point of the baseline. At 80% the gap is 1.28 points, just outside the budget, and at 90% it is 3.36 points. With only 10–20% of the weights left, 5 epochs are not enough to recover fully, or the remaining capacity is simply too small.

### [QUESTION 1] Which model would you keep and why?

**Kept model: 70% global sparsity + retraining**, with **68.05%** validation accuracy (0.62 points below the baseline). File: `exercise4_finetuned_70.pt`.

**Why:**
- **It is the most compressed model that meets the 1-point rule.** It keeps only **30% of the weights** (about 846k non-zero out of 2.82M), against 50% for the best model without retraining (Exercise 3). So retraining allows **20 percentage points more sparsity for the same accuracy budget**.
- **It is still well inside the budget** (−0.62 point). The 60% model is closer to the baseline (68.52%, −0.15), but it keeps a third more weights for a gain smaller than the noise level.
- **The 80% model (67.39%, −1.28) narrowly misses the rule.** It would be a good choice if a budget of about 1.5 points were accepted: it keeps only 20% of the weights. Longer retraining or the iterative approach (Exercise 5) might bring it within 1 point.
- **The 90% model (−3.36 points)** loses too much for this budget.

**Practical note, as in Exercise 1:** this sparsity is **unstructured**, so the saved file is not smaller and inference is not faster on standard hardware without sparse storage or sparse kernels. The gain here is in the number of non-zero weights.

**Limitations:**
- These are single-run results on the validation set. Small differences (< 0.3 points) are not meaningful.
- Only one retraining setting was tested (5 epochs, lr 0.01).
- The final chosen model will be evaluated once on the **test set** after Exercise 5, so that test data is not used for model selection.

## 5. Conclusion

Exercise 4 is complete:
- each of the 12 Exercise 3 models was retrained for 5 epochs at a 10× lower learning rate, with the masks kept active (sparsity checked after training);
- every model was evaluated on the same validation split;
- the accuracy / sparsity curve before vs after retraining was drawn, and a model was chosen.

**Main result:** **pruning followed by short retraining keeps the accuracy within 1 point of the baseline up to 70% sparsity**, against 50% without retraining. Even the 90% model recovers from 4.6% to 65.3%. **Kept model: 70% sparsity, 68.05%** validation accuracy.

## 6. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output: settings, each epoch of each model, before/after, sparsity check, chosen model |
| `exercise4_results.csv` | Per ratio: sparsity, val accuracy/loss before and after, gain, gap to baseline |
| `exercise4_history.csv` | Every epoch of every model (lr, train/val loss and accuracy) |
| `exercise4_curve.png/.pdf` | Accuracy and loss vs sparsity, before vs after retraining |
| `exercise4_retraining_history.png/.pdf` | Validation accuracy during retraining (50–90%) |
| `exercise4_finetuned_00.pt` … `exercise4_finetuned_90.pt` | The 12 retrained models (with masks) |
