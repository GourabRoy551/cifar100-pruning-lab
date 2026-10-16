# Exercise 3: Global pruning

**Code:** `pruning.py`, Exercise 3 block (`run_exercises = [3]`)
**Pruning method:** global L1 unstructured pruning, `prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=ratio)`, applied to the weights and biases of every layer, **without retraining**
**Evaluation data:** validation set (10,000 images), the same saved split (`cifar100_split.pt`) and pre-processing as for training

## 1. Task (from the template)

Instead of pruning each layer separately (Exercises 1–2), prune the **whole network at once** by removing the lowest weights of the entire model with `global_unstructured`. It takes three arguments:
1. `parameters`: the list of (layer, weight name) tuples. It is the same list as in Exercise 1: all weights and biases of all conv, BatchNorm and linear layers, 62 tensors.
2. `pruning_method`: `prune.L1Unstructured`.
3. `amount`: the sparsity ratio.

**Difference from layer-by-layer pruning:**
- **Layer by layer:** removes x% **of each layer**.
- **Global:** all weights of the network are ranked together by absolute value, and the x% smallest are removed **wherever they are**. This is the same as applying one **global threshold**: every weight with |w| ≤ threshold becomes 0. Each layer therefore ends up with a different sparsity.

**Same protocol as Exercise 2:**
- same 12 ratios (0–90%);
- a fresh model with the original weights for every ratio;
- the same validation split;
- the same rule for choosing a model: **the highest sparsity whose validation accuracy is at most 1 point below the baseline**, i.e. at least 67.67%.

As the template suggests, every pruned model was saved **with its masks** (`weight_orig` + `weight_mask`), so that Exercise 4 can retrain them while keeping the pruned weights at zero.

## 2. Results

Baseline (unpruned): **68.67%** validation accuracy, loss 1.1873.

| Sparsity ratio | Global threshold (pruned if \|w\| ≤) | Val. accuracy (global) | Drop (points) | Val. loss | Val. accuracy, layer by layer (Ex. 2) |
|---:|---:|---:|---:|---:|---:|
| 0% | — | 68.67% | 0.00 | 1.1873 | 68.67% |
| 5% | 0.0013 | 68.67% | 0.00 | 1.1871 | 64.66% |
| 10% | 0.0026 | 68.77% | −0.10 | 1.1874 | 55.20% |
| 15% | 0.0039 | 68.79% | −0.12 | 1.1878 | 41.57% |
| 20% | 0.0053 | 68.61% | 0.06 | 1.1885 | 26.73% |
| 30% | 0.0081 | 68.78% | −0.11 | 1.1909 | 7.22% |
| 40% | 0.0111 | 68.21% | 0.46 | 1.2066 | 3.35% |
| **50%** | **0.0144** | **67.67%** | **1.00** | **1.2195** | 1.03% |
| 60% | 0.0182 | 65.09% | 3.58 | 1.2940 | 0.99% |
| 70% | 0.0228 | 58.22% | 10.45 | 1.5543 | 1.17% |
| 80% | 0.0291 | 34.60% | 34.07 | 2.7524 | 1.04% |
| 90% | 0.0397 | 4.61% | 64.06 | 5.7027 | 0.95% |

The actual sparsity equals the target at every ratio. A negative drop means a slightly higher accuracy than the baseline.

**Figures:**
- `exercise3_curve.png/.pdf`: accuracy and loss vs sparsity, global pruning compared with layer-by-layer pruning (Ex. 2).
- `exercise3_layer_sparsity.png/.pdf`: sparsity of each layer of the kept model (50%).

## 3. Answers

### [QUESTION] Draw the accuracy / sparsity ratio curve and comment on the results.

**Global pruning is much better than layer-by-layer pruning at every sparsity level:**

| Sparsity | Layer by layer (Ex. 2) | Global (Ex. 3) |
|---:|---:|---:|
| 20% | 26.73% | 68.61% |
| 50% | 1.03% (chance) | 67.67% |
| 70% | 1.17% (chance) | 58.22% |

**The global curve has three regions:**

1. **0 → 50%: almost no loss.** Accuracy stays within ±1 point of the baseline. Up to 30% it is practically unchanged (68.61–68.79%), and the loss barely moves (1.187 → 1.191).
   - The small "gains" at 10%, 15% and 30% (+0.10 to +0.12 points) are **noise**, not a real improvement: 0.1 point is only 10 images out of 10,000. The loss, which is more sensitive, increases very slightly but steadily, so the model does not get better.
   - **Half of the weights can be removed** with at most 1 point of loss, without any retraining. Most of the weights of this trained network are very small (the threshold at 50% is only |w| ≤ 0.0144) and contribute almost nothing to the output.
2. **50 → 70%: gradual decline.** −3.58 points at 60%, −10.45 points at 70%. The removed weights are now large enough to matter.
3. **70 → 90%: collapse.** 34.60% at 80%, and 4.61% at 90%, close to chance. At 90% the loss (5.70) is above ln(100) ≈ 4.61, the loss of a uniform guess, so the predictions are confidently wrong, as in the destroyed models of Exercise 2.

**Why global pruning works so much better:**
- **One threshold for the whole network.** Global pruning decides *where* to prune: every layer loses only the weights that are small **compared with the rest of the network**.
- **Different layers have different weight scales.** The large layers (e.g. stage 4, 589,824 weights each) contain many very small weights and lose many of them. The small, sensitive layers (first conv, BatchNorm, classifier) have larger weights and are mostly spared (see the next question).
- **Layer-by-layer pruning forces the same fraction on every layer,** which damages the sensitive layers first.

**Which model is kept:**
- **Rule result: 50% sparsity**, with 67.67% validation accuracy.
- **Caveat: 50% is exactly on the limit** of the rule (−1.00 point). The difference between 50% and 40% is small and comes from a single run, so **40%** (68.21%, −0.46 point) is a safer alternative with a margin below the 1-point budget.
- **Above 50%,** the losses grow quickly (−3.6 points at 60%, −10.5 at 70%).
- **For this unretrained network, the useful range is up to about 40–50% sparsity.** Exercise 4 tests whether retraining can push this limit further.

### [QUESTION] For the best model, what is the sparsity ratio of each layer?

**Best model:** global pruning at **50%**. The total is exactly 50.00%, but the layers have **very different** sparsities, from **0% to 64.44%**.

**By layer type:**

| Layer type | Zeros / parameters | Sparsity |
|---|---:|---:|
| Conv2d (20 layers) | 1,407,750 / 2,790,240 | **50.45%** |
| BatchNorm2d (20 layers) | 109 / 4,800 | **2.27%** |
| Linear (classifier) | 2,511 / 25,700 | **9.77%** |

**Every layer:**

| Layer | Type | Parameters | Sparsity |
|---|---|---:|---:|
| stem.0 | Conv2d | 864 | 9.26% |
| stem.1 | BatchNorm2d | 64 | 3.12% |
| layer1.0.conv1 | Conv2d | 9,216 | 37.70% |
| layer1.0.bn1 | BatchNorm2d | 64 | 1.56% |
| layer1.0.conv2 | Conv2d | 9,216 | 33.57% |
| layer1.0.bn2 | BatchNorm2d | 64 | 6.25% |
| layer1.1.conv1 | Conv2d | 9,216 | 33.91% |
| layer1.1.bn1 | BatchNorm2d | 64 | 10.94% |
| layer1.1.conv2 | Conv2d | 9,216 | 31.45% |
| layer1.1.bn2 | BatchNorm2d | 64 | 4.69% |
| layer2.0.conv1 | Conv2d | 18,432 | 34.50% |
| layer2.0.bn1 | BatchNorm2d | 128 | 3.12% |
| layer2.0.conv2 | Conv2d | 36,864 | 34.69% |
| layer2.0.bn2 | BatchNorm2d | 128 | 7.03% |
| layer2.0.shortcut.0 | Conv2d (1×1) | 2,048 | 13.72% |
| layer2.0.shortcut.1 | BatchNorm2d | 128 | 7.03% |
| layer2.1.conv1 | Conv2d | 36,864 | 32.23% |
| layer2.1.bn1 | BatchNorm2d | 128 | 0.00% |
| layer2.1.conv2 | Conv2d | 36,864 | 34.97% |
| layer2.1.bn2 | BatchNorm2d | 128 | 3.91% |
| layer3.0.conv1 | Conv2d | 73,728 | 33.61% |
| layer3.0.bn1 | BatchNorm2d | 256 | 4.30% |
| layer3.0.conv2 | Conv2d | 147,456 | 37.95% |
| layer3.0.bn2 | BatchNorm2d | 256 | 7.81% |
| layer3.0.shortcut.0 | Conv2d (1×1) | 8,192 | 20.51% |
| layer3.0.shortcut.1 | BatchNorm2d | 256 | 8.20% |
| layer3.1.conv1 | Conv2d | 147,456 | 43.45% |
| layer3.1.bn1 | BatchNorm2d | 256 | 0.00% |
| layer3.1.conv2 | Conv2d | 147,456 | 47.54% |
| layer3.1.bn2 | BatchNorm2d | 256 | 2.73% |
| layer4.0.conv1 | Conv2d | 294,912 | 45.20% |
| layer4.0.bn1 | BatchNorm2d | 512 | 0.20% |
| layer4.0.conv2 | Conv2d | 589,824 | 53.43% |
| layer4.0.bn2 | BatchNorm2d | 512 | 0.00% |
| layer4.0.shortcut.0 | Conv2d (1×1) | 32,768 | 27.28% |
| layer4.0.shortcut.1 | BatchNorm2d | 512 | 0.00% |
| layer4.1.conv1 | Conv2d | 589,824 | 50.32% |
| layer4.1.bn1 | BatchNorm2d | 512 | 0.00% |
| layer4.1.conv2 | Conv2d | 589,824 | **64.44%** (highest) |
| layer4.1.bn2 | BatchNorm2d | 512 | 0.98% |
| fc | Linear | 25,700 | 9.77% |

The same values are in `exercise3_layer_sparsity.csv` and the bar chart `exercise3_layer_sparsity.png`.

**Observations:**
1. **The deeper and larger the layer, the more it is pruned.**
   - Stage 1 convs (9,216 weights): 31–38%.
   - Stage 2: 32–35%.
   - Stage 3: 34–48%.
   - Stage 4 (295k–590k weights): 45–64%.
   - The last conv, `layer4.1.conv2`, alone loses 64% of its weights. The large layers are the most redundant, and global pruning finds this by itself.
2. **The first conv (`stem.0`) is mostly kept (9.26%).** It works directly on the image with only 864 weights, so each weight matters, and its weights are larger than the global threshold.
3. **The 1×1 shortcut convs are pruned less** (13.7%, 20.5%, 27.3%) than the 3×3 convs of the same stage. They carry the residual signal to the next stage.
4. **The classifier (`fc`) is mostly kept (9.77%).** Its weights map the features directly to the 100 classes.
5. **BatchNorm is almost untouched (2.27% overall,** several layers at 0%), against about 20% in **every** BatchNorm layer with layer-by-layer pruning at 20% (Exercise 1).
   - The BatchNorm scales γ are generally much larger than the threshold (0.0144). The few BatchNorm zeros are most likely shifts β that were already close to zero. Weights and biases are counted together here, so this split was not measured.
   - Not pruning BatchNorm means that no channel is switched off, which is very likely an important reason why global pruning keeps the accuracy while layer-by-layer pruning loses it. This was not tested in a separate experiment.
6. **Why the large layers have smaller weights:** with Kaiming initialisation, the initial weight scale is about √(2 / fan), so layers with more inputs/outputs start with smaller weights. Training (with weight decay) keeps many of them small. A single global threshold therefore removes more in wide layers.
   - **Limitation of global magnitude pruning:** it compares weights from layers with different scales, so "small" partly means "belongs to a large layer". Here this works well, because those large layers are also the most redundant.

## 4. Conclusion

Exercise 3 is complete:
- global L1 pruning was applied with `global_unstructured` at 12 sparsity ratios, reloading the original weights each time;
- all models were evaluated on the same validation split and saved with their masks for Exercise 4;
- the accuracy / sparsity curve was drawn and compared with Exercise 2, and the per-layer sparsity of the best model was measured.

**Main results:**
- **Global pruning keeps the accuracy within 1 point up to 50% sparsity** (67.67%; 68.21% at 40%), **without retraining**. Layer-by-layer pruning already loses 4 points at 5%, and reaches chance level at 50%.
- **Kept model: 50% sparsity**, exactly at the 1-point limit; 40% is the safer choice.
- **Global pruning distributes the sparsity unevenly and sensibly:**
  - large deep convs are pruned the most (up to 64%);
  - the first conv (9%), the classifier (10%) and BatchNorm (2%) are mostly kept.

## 5. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output of the run (results, thresholds, per-layer and per-type sparsity) |
| `exercise3_results.csv` | Ratio, actual sparsity, threshold, val loss, val accuracy, drop, model file, one row per ratio |
| `exercise3_layer_sparsity.csv` | Per-layer sparsity of the kept model (50%) |
| `exercise3_curve.png/.pdf` | Global vs layer-by-layer accuracy and loss curves |
| `exercise3_layer_sparsity.png/.pdf` | Per-layer sparsity of the kept model |
| `exercise3_global_00.pt` … `exercise3_global_90.pt` | The 12 globally pruned models, **with masks** (input of Exercise 4) |
