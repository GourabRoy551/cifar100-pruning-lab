# Exercise 2: Accuracy / sparsity ratio curve

**Code:** `pruning.py`, Exercise 2 block (`run_exercises = [2]`)
**Pruning method:** layer-by-layer L1 unstructured pruning (`l1_unstructured`, weights and biases of every layer), the same as Exercise 1, **without retraining**
**Evaluation data:** validation set (10,000 images), the **same saved train/val split** (`cifar100_split.pt`) and pre-processing as for training the baseline

## 1. Task (from the template)

To find the best compression ratio, prune the network with different sparsity ratios and evaluate each pruned model on the validation set.

The template's tips, and how each was followed:

| Tip | How it was done |
|---|---|
| Use the same train/val split as for training | The split is loaded from `cifar100_split.pt`, never regenerated |
| Move the Exercise 1 code into a function | `layerwise_pruning(model, sparsity_ratio)` |
| Reload the original weights before each pruning | `load_model()` builds a new `ConvNet` and loads `best_model_params.pt` for **every** ratio |
| Save the (sparsity ratio, validation accuracy) pairs and the pruned weights | `exercise2_results.csv` is rewritten after every ratio, and each pruned model is saved (`exercise2_layerwise_XX.pt`, after `prune.remove`) |

**Sparsity ratios tested:** 0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80 and 90%. Steps are finer below 20% because Exercise 1 showed accuracy already collapses at 20%.

**Rule for choosing a model** (used in every exercise): keep the **highest sparsity whose validation accuracy is at most 1 percentage point below the baseline**, i.e. at least 67.67%.

## 2. Results

Baseline (unpruned): **68.67%** validation accuracy, loss 1.1873.

| Sparsity ratio | Actual sparsity | Val. accuracy | Drop (points) | Val. loss |
|---:|---:|---:|---:|---:|
| 0% | 0.00% | **68.67%** | 0.00 | 1.1873 |
| 5% | 5.00% | 64.66% | 4.01 | 1.3270 |
| 10% | 10.00% | 55.20% | 13.47 | 1.6845 |
| 15% | 15.00% | 41.57% | 27.10 | 2.3515 |
| 20% | 20.00% | 26.73% | 41.94 | 3.3057 |
| 30% | 30.00% | 7.22% | 61.45 | 5.2176 |
| 40% | 40.00% | 3.35% | 65.32 | 5.4980 |
| 50% | 50.00% | 1.03% | 67.64 | 5.6951 |
| 60% | 60.00% | 0.99% | 67.68 | 5.5438 |
| 70% | 70.00% | 1.17% | 67.50 | 5.3133 |
| 80% | 80.00% | 1.04% | 67.63 | 4.9617 |
| 90% | 90.00% | 0.95% | 67.72 | 5.2010 |

**Figure:** `exercise2_curve.png/.pdf`, with validation accuracy vs sparsity on the left (baseline and chance lines shown, chosen ratio marked) and validation loss vs sparsity on the right.

**Consistency checks:**
- At 0% the accuracy equals the baseline exactly, as expected, since nothing is removed.
- At 20% the result equals Exercise 1 exactly (26.73%, loss 3.3057). This confirms that the original weights were correctly reloaded before each pruning.
- The actual sparsity equals the target at every ratio.

## 3. Answers

### [QUESTION] Comment on the obtained curve.

The curve has three regions:

1. **0 → 5%: small loss.** Removing the 5% smallest weights in every layer costs 4 points (68.67% → 64.66%). The smallest weights contribute little, but even 5% is not free for this network.
2. **5 → 30%: collapse.** Accuracy falls very fast, losing about 10–15 points for every extra 5% of sparsity: 55.20% at 10%, 41.57% at 15%, 26.73% at 20%, and only 7.22% at 30%.
3. **≥ 40–50%: the network is destroyed.** From 50% on, accuracy sits at the **chance level for 100 classes (about 1%)**: 0.95–1.17%. The small differences between 50% and 90% are just noise around chance.

**The loss curve says the same thing:**
- It rises from 1.19 to about 5.2 at 30%, then stays between 5.0 and 5.7.
- These values are **above ln(100) ≈ 4.61**, the loss of a model that guesses uniformly. So the heavily pruned networks are not merely uncertain: they make **confident wrong predictions**, as expected once the internal representations are broken.

**Why layer-by-layer pruning fails so fast on this model:** layer-by-layer pruning removes the **same fraction from every layer**, whatever the layer's size or importance. This was not tested separately here, but three likely reasons are:
- **Small, sensitive layers** (first conv: 864 weights; classifier: 25,700 weights; every BatchNorm layer) are pruned as hard as the large, redundant stage-4 convs (589,824 weights each).
- **BatchNorm scales.** Each BatchNorm scale γ multiplies a whole channel, so pruning x% of the γ values switches off about x% of the channels in **every** layer. The BatchNorm parameters are only 0.17% of the model, but this effect is much stronger than removing single conv weights.
- **The model is compact.** It has 2.8M parameters, a quarter of a standard ResNet18, so it has little redundancy to spare.

This also shows that **"smallest weights are unimportant" only holds within limits**. The L1 criterion ranks weights *inside* each layer, but it cannot tell that a layer is more sensitive than another. Global pruning (Exercise 3) addresses exactly this.

### [QUESTION] Which sparsity ratio would you keep and why?

**Kept ratio: 0%, i.e. no pruning**, with this method and without retraining.

**Justification:**
- **None of the pruned models meets the 1-point rule.** The lowest pruning tested, 5%, already costs **4.01 points**, four times the allowed loss.
- **Pruning gains little here anyway.** Unstructured zeros do not reduce the file size or the inference time (see Exercise 1: the file stays at 11.34 MB). Paying 4 points of accuracy for 5% zeros is not a good trade.
- **If a larger loss were acceptable,** for example a 5-point budget, then **5%** would be the only sensible choice (64.66%). From 10% on, the model loses at least 13 points, and the curve shows each further step costs much more.

**What this tells us:** one-shot **layer-by-layer** pruning is a poor way to compress this network. A useful nonzero ratio needs either:
- a smarter allocation of sparsity across layers (**global pruning, Exercise 3**), or
- **retraining after pruning** (Exercises 4 and 5), so the remaining weights can make up for the removed ones.

**Limitations:**
- The ratios between 0% and 5% (e.g. 1–4%) were not tested, so a very small nonzero ratio might still meet the 1-point rule. Such tiny compression would bring no practical benefit.
- All results come from a single trained model and are measured on the validation set only. The test set is kept for the final chosen model.

## 4. Conclusion

Exercise 2 is complete:
- the Exercise 1 pruning was turned into a function and applied at 12 sparsity ratios;
- the original weights were reloaded before each ratio;
- every pruned model was evaluated on the same validation split used for training;
- the results and pruned weights were saved, and the accuracy / sparsity curve was drawn.

**Answer:** with layer-by-layer pruning and no retraining, accuracy drops by 4 points at 5% sparsity, collapses between 5% and 30%, and reaches chance level from 50%. Under the 1-point rule, **0% is the ratio to keep**.

## 5. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output of the run |
| `exercise2_results.csv` | Sparsity ratio, actual sparsity, val loss, val accuracy, drop, model file, one row per ratio |
| `exercise2_curve.png/.pdf` | Accuracy / sparsity and loss / sparsity curves |
| `exercise2_layerwise_00.pt` … `exercise2_layerwise_90.pt` | The 12 pruned models (after `prune.remove`) |
