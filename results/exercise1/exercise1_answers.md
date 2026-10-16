# Exercise 1: Pruning a pre-trained model

**Code:** `pruning.py`, Exercise 1 block (`run_exercises = [1]`)
**Model pruned:** Exercise 0 baseline, `best_model_params.pt` (ResNet18-like ConvNet, 2,820,740 parameters)
**Evaluation data:** validation set (10,000 images), the same saved split (`cifar100_split.pt`) and pre-processing used for training

## 1. Task (from the template)

Prune the pre-trained model with:
- **granularity:** unconstrained (unstructured, individual weights)
- **scope:** layer by layer
- **weight selection:** L1 norm (smallest absolute values)

The template's steps:
1. Retrieve a list of all layers.
2. For each layer, get its different weights.
3. Prune them with `torch.nn.utils.prune.l1_unstructured`, biases included, for a given sparsity ratio (20%).
4. Look at which entries were kept, using `model.named_buffers()`.
5. "Compress" the model file by permanently removing the original weights with `prune.remove`.

## 2. What was done

1. **Load.** A fresh `ConvNet` was built and the trained weights were loaded. Baseline validation accuracy: **68.67%**.
2. **Collect.** All layers holding parameters were collected: **41 layers** (20 Conv2d, 20 BatchNorm2d, 1 Linear), giving **62 tensors** (every weight and every existing bias). The conv layers have no bias (`bias=False`, as is usual before BatchNorm), so the biases pruned are those of the BatchNorm and Linear layers.
3. **Prune.** `prune.l1_unstructured(layer, name, amount=0.2)` was called on each tensor separately. In each tensor, the 20% of entries with the smallest absolute value are set to zero.
4. **Inspect.** PyTorch does not delete weights. Each pruned tensor `weight` becomes `weight_orig` (the original values, still a parameter) multiplied by `weight_mask` (a 0/1 buffer). **62 masks** were created, one per tensor. For example, the mask of the first 3×3 kernel of the first conv was:
   ```
   [1, 1, 1]
   [1, 1, 0]      1 = kept, 0 = pruned
   [1, 1, 1]
   ```
5. **Evaluate** the pruned model on the validation set.
6. **Compress.** The masked model was saved, then `prune.remove(layer, name)` was called on every tensor, and the model was saved again.

## 3. Results

| | Baseline | Pruned 20% (layer by layer) |
|---|---:|---:|
| Zero parameters | 0 | 564,148 / 2,820,740 |
| Actual sparsity | 0% | **20.00%** |
| Validation accuracy | **68.67%** | **26.73%** |
| Validation loss | 1.1873 | 3.3057 |
| Accuracy drop | | **−41.94 points** |
| Accuracy after `prune.remove` | | 26.73% (identical) |

| Saved file | Size |
|---|---:|
| Baseline `best_model_params.pt` | 11.34 MB |
| Pruned, with masks (`exercise1_pruned_20_masked.pt`) | 22.65 MB |
| Pruned, after `prune.remove` (`exercise1_pruned_20.pt`) | 11.34 MB |

### Figures

- `exercise1_layer_sparsity.png/.pdf`: sparsity of each of the 41 layers.
- `exercise1_weight_histogram.png/.pdf`: weight distribution of `layer3.0.conv2`, with the pruned band shaded. The band is |w| ≤ 0.0073, the 20% smallest values of that layer.

## 4. Answers and discussion

### Q1. Was the network pruned as intended (layer by layer, 20%, L1)?

Yes.
- **Overall:** the total sparsity is exactly **20.00%** (564,148 zeros out of 2,820,740).
- **Every layer is at about 20%** (see `exercise1_layer_sparsity.png`). This is what defines layer-by-layer pruning: each layer loses the **same fraction** of its own weights, whatever its size or importance.
- **Small deviations come from rounding.** PyTorch prunes `round(0.2 × n)` entries in each tensor. A 32-element BatchNorm tensor loses 6 entries (18.75%), a 128-element one loses 26 (20.31%) and a 512-element one loses 102 (19.92%). Large conv layers are at 20.00%.
- **The removed weights are the smallest ones.** The histogram shows that L1 pruning removes the weights in a narrow band around zero. In `layer3.0.conv2`, every weight with |w| ≤ 0.0073 was removed, which is the central 20% of the distribution.

### Q2. Which entries are kept? (template tip: `model.named_buffers()`)

- **Masks are buffers.** After pruning, `model.named_buffers()` contains a `<name>_mask` buffer for every pruned tensor, for example `stem.0.weight_mask` with shape (32, 3, 3, 3), or `layer1.0.bn1.weight_mask` with shape (32,).
- **Reading a mask:** a 1 marks a kept index and a 0 a pruned one. In the example kernel above, 1 of 9 weights was pruned. Across that whole layer, 79.98% of the entries are kept.
- **The mask is fixed:** it stays the same until it is removed or pruned again.
- **What the model actually uses:** in every forward pass, PyTorch recomputes `weight = weight_orig * weight_mask`.

### Q3. Effect of pruning on accuracy

- **The drop is large.** Validation accuracy falls from **68.67% to 26.73%** (−41.94 points), and the loss nearly triples (1.19 → 3.31). The model still does much better than chance (1% for 100 classes), but it has lost about 60% of its correct predictions.
- **Possible explanation, untested:** layer-by-layer pruning treats all layers the same, and several factors could make that harmful. None was tested separately in this exercise:
  - **Sensitive small layers lose 20% too.** The first conv (864 weights) and the classifier (25,700 weights) are pruned as hard as the large redundant layers in stage 4 (589,824 weights each).
  - **BatchNorm parameters are pruned at 20%.** They are only 4,800 parameters (0.17% of the model), but each BatchNorm scale γ multiplies a **whole channel**. Setting the 20% smallest γ to zero in every BatchNorm switches off about 20% of the channels in every layer, so the change is closer to channel pruning than to removing single weights. Pruning the BatchNorm shifts β also changes the activation statistics.
  - **The model is compact.** At 2.8M parameters (a quarter of a standard ResNet18), it probably has less redundancy to lose. In an earlier run with the full-width ResNet18 (11.2M parameters), the same 20% layer-by-layer pruning cost about 27 points instead of 42.
- **Verifying the cause** would need a separate test, for example pruning conv/linear only versus BatchNorm only. Exercise 3 (global pruning) gives an indirect comparison.
- **This is a valid result, not a bug.** The accuracy after `prune.remove` is identical (26.73%), which confirms that removing the re-parametrization keeps exactly the same pruned weights.
- **No fine-tuning yet.** These are one-shot results without retraining. Exercises 4 and 5 show how much retraining can recover.

### Q4. Does pruning compress the model file? (`prune.remove`)

- **With masks, the file doubles:** 11.34 MB → 22.65 MB. It stores both `weight_orig` (all original values) and `weight_mask` (one entry per weight).
- **After `prune.remove`, it is back to 11.34 MB**, the same as the baseline. `prune.remove` makes the pruning permanent: it writes the zeros into `weight` and deletes `weight_orig` and `weight_mask`.
- **The file is not smaller,** because the zeros are still stored as ordinary 32-bit numbers in dense tensors, and the tensor shapes and parameter count do not change.
- **Saving real memory or compute** would need sparse storage formats, compression of the zeros (e.g. zip), or **structured** pruning that removes whole filters or channels (see the Bonus exercise).
- **Conclusion:** unstructured pruning here reduces the number of non-zero weights, not the file size or the inference time.

## 5. Conclusion

Exercise 1 is complete and does what the template asks:
- the model is pruned with `l1_unstructured`, layer by layer, at 20%, biases included;
- the masks were inspected with `named_buffers()`;
- the pruning was made permanent with `prune.remove`.

The main observation is that **20% layer-by-layer pruning, without retraining, costs about 42 accuracy points** on this compact model, even though only the smallest-magnitude weights were removed. This motivates the next exercises: measuring accuracy for several sparsity ratios (Ex. 2), letting the network choose which layers to prune (global pruning, Ex. 3), and retraining after pruning (Ex. 4–5).

## 6. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output of the run |
| `exercise1_results.csv` | Main numbers (sparsity, accuracies, losses, file sizes) |
| `exercise1_layer_sparsity.csv` | Parameters, zeros and sparsity of each layer |
| `exercise1_layer_sparsity.png/.pdf` | Per-layer sparsity plot |
| `exercise1_weight_histogram.png/.pdf` | Weight distribution of `layer3.0.conv2` and the pruned band |
| `exercise1_pruned_20_masked.pt` | Pruned model with `weight_orig` + `weight_mask` |
| `exercise1_pruned_20.pt` | Pruned model after `prune.remove` (plain weights with zeros) |
