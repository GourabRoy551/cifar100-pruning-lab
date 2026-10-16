# Bonus Exercise: Custom weight selection function (structured pruning)

**Code:** `pruning.py`, Bonus section (`run_exercises = ["bonus"]`): class `MyPruningMethod`, functions `structured_pruning()` and `count_zero_filters()`, reusing `fine_tune()` from Exercise 4
**Evaluation data:** validation set (10,000 images), the same saved split and pre-processing as for training

## 1. Task (from the template)

Instead of using a provided function for the weight selection, implement our own pruning method by inheriting from `torch.nn.utils.prune.BasePruningMethod`. It should use **structured granularity**: remove the `amount` lowest "columns" (by L1 norm) of the weights with dimensionality > 2.

**Interpretation used:** for a conv weight of shape (out_channels × in_channels × k × k), the pruned structures are **whole filters, i.e. output channels (dimension 0)**.
- This is the standard form of structured pruning, and the one that can actually make a network smaller: a zero filter means a zero output channel, which can be removed together with the matching input channel of the next layer.
- The alternative reading (input channels, dimension 1) was not implemented.

## 2. Implementation

**`MyPruningMethod(prune.BasePruningMethod)`**, which completes the class skeleton given in the template:
- `PRUNING_TYPE = 'structured'`, `dim = 0`, and an `amount` parameter: the fraction of filters to remove.
- **`compute_mask(t, default_mask)`:**
  1. If the tensor has ≤ 2 dimensions (linear weights, BatchNorm parameters, biases), the mask is returned unchanged, so nothing is pruned, as the template specifies.
  2. Otherwise, compute the **L1 norm of each filter**: the sum of |w| over its input channels and kernel positions. Any earlier mask (`default_mask`) is taken into account.
  3. Find the `round(amount × number of filters)` filters with the **smallest L1 norm** (`torch.topk(..., largest=False)`) and set their **whole mask slice to 0**.
- It is used like the built-in methods: `MyPruningMethod.apply(layer, "weight", amount)`. This creates `weight_orig` + `weight_mask`, exactly like `l1_unstructured`.

**Experiment:**
- **Pruning:** `structured_pruning(model, amount)` applies the method to the weight of **every conv layer**, layer by layer: the same fraction of filters in each of the 20 conv layers (2,400 filters in total).
- **Filter ratios:** 0, 10, 20, 30, 40 and 50%. A **fresh model with the original weights** is loaded for every ratio.
- **Retraining:** each pruned model is evaluated, then retrained with `fine_tune()` (5 epochs, lr 0.01 with cosine decay, masks kept active, as in Exercise 4) and evaluated again.
- **Rule for choosing a model:** the same as before, i.e. the highest ratio with validation accuracy at most 1 point below the baseline (≥ 67.67%).

**Checks:**
- **Exact filter counts:** the number of entirely zero filters is exactly the expected one (240, 480, 720, 960, 1,200 of 2,400).
- **Zeros survive retraining:** the counts are the same **after retraining**, so the pruned filters stayed zero.
- **The right filters are removed:** a test on the baseline confirmed that every removed filter has a lower L1 norm than every kept filter of the same layer.

## 3. Results

Baseline (unpruned): **68.67%** validation accuracy, loss 1.1873.

| Filters removed per conv layer | Zero filters | Actual sparsity (all parameters) | Val. acc. right after pruning | Val. acc. after retraining | Gap to baseline (points) | Val. loss after |
|---:|---:|---:|---:|---:|---:|---:|
| 0% (control) | 0 / 2400 | 0.00% | 68.67% | 68.42% | 0.25 | 1.2346 |
| 10% | 240 / 2400 | 10.00% | 55.06% | 68.42% | 0.25 | 1.2251 |
| **20%** | **480 / 2400** | **19.78%** | 18.37% | **67.76%** | **0.91** | 1.2230 |
| 30% | 720 / 2400 | 29.68% | 2.70% | 67.45% | 1.22 | 1.2187 |
| 40% | 960 / 2400 | 39.46% | 1.64% | 66.40% | 2.27 | 1.2236 |
| 50% | 1200 / 2400 | 49.46% | 1.87% | 65.08% | 3.59 | 1.2647 |

The actual sparsity is slightly below the filter ratio because the BatchNorm and linear parameters are not pruned (≤ 2 dimensions), and because of rounding in small layers.

### Comparison with unstructured global pruning at the same sparsity

| Sparsity | Unstructured, no retraining (Ex. 3) | Structured, no retraining | Unstructured + retraining (Ex. 4) | Structured + retraining |
|---:|---:|---:|---:|---:|
| 10% | 68.77% | 55.06% | 68.35% | 68.42% |
| 20% | 68.61% | 18.37% | 68.51% | 67.76% |
| 30% | 68.78% | 2.70% | 68.40% | 67.45% |
| 40% | 68.21% | 1.64% | 68.66% | 66.40% |
| 50% | 67.67% | 1.87% | 68.41% | 65.08% |

**Figure:** `bonus_curve.png/.pdf` shows accuracy and loss against the actual sparsity for structured and unstructured pruning, before and after retraining.

## 4. Discussion

**1. Without retraining, structured pruning is far more damaging than unstructured pruning.**
- Removing 10% of the filters already costs 13.6 points (55.06%). At 20% the model is down to 18.37%, and from 30% it is at chance level (about 2%).
- At the same sparsity, unstructured global pruning (Ex. 3) loses nothing (68.6–68.8%).
- **Why:** unstructured pruning removes the individually smallest weights, scattered everywhere. Structured pruning must remove **whole filters**, so it also removes many large, useful weights that belong to low-norm filters. Each removed filter deletes an **entire feature map** (output channel) for every following layer.
- **Coincidence worth noting:** the 10% structured result (55.06%) is almost identical to 10% layer-by-layer unstructured pruning in Exercise 2 (55.20%). Both remove the same fraction in every layer, the setting already shown to be harmful.

**2. Retraining recovers most of the accuracy.**
- **10%:** after 5 epochs, the model reaches 68.42%, **exactly the same as the retrained unpruned control** (68.42%). Removing 10% of the filters costs nothing once the network is retrained.
- **20%:** **67.76%** (−0.91 point), within the 1-point rule.
- **30–50%:** 67.45%, 66.40% and 65.08%. The network recovers from chance level to within 1.2–3.6 points of the baseline.
- **Fast recovery:** as in Exercise 4, most of it happens in the first epoch. For example, at 30%, accuracy goes from 2.70% to 62.79% after one epoch.

**3. Structured pruning is still less accurate than unstructured pruning after retraining.**
- **The gap grows with sparsity:** −0.75 point at 20%, −0.95 at 30%, −2.26 at 40%, −3.33 at 50%, compared with unstructured global + retraining (Ex. 4).
- **This is the expected price of structure:** the pruning is less free to choose which weights to remove.

**4. But structured zeros are useful zeros.**
- **This is the main advantage of the bonus method.** The unstructured models of Exercises 1–5 keep the same tensor shapes, so they are neither smaller nor faster without sparse storage or kernels (see Exercise 1).
- **Here, 20–50% of the filters of every conv layer are entirely zero.** These channels can be **physically removed**: delete the zero filters and the matching input-channel slices of the next conv layer.
- **What this would give:** a genuinely smaller dense network that is faster on any hardware.
- **Size of the gain (estimate, not measured):** when both the input and output channels of a conv layer are reduced by 20%, its parameters and computations shrink to about 0.8 × 0.8 = 64% of the original.
- **Not done in this lab (would require extra channel bookkeeping):**
  - **Residual connections:** the output channels of a residual block are added to the shortcut, so they must be removed consistently on both paths. The internal channels of each block (after `conv1`) can be removed freely.
  - **BatchNorm shift:** a zeroed filter's BatchNorm still outputs its constant shift β. That constant would have to be folded into the next layer's bias when the channel is removed.

**5. The loss after retraining behaves as in Exercise 4.**
- The validation loss after retraining stays around 1.22, similar to or slightly lower than the retrained control (1.23).
- Training accuracy decreases with the pruning ratio (94.4% at 0%, 78.2% at 50%), so the pruned networks overfit less.

## 5. Which model would be kept?

**Kept model: 20% of the filters removed + retraining**, with **67.76%** validation accuracy (−0.91 point, within the 1-point rule). File: `bonus_structured_20.pt`.
- **What it removes:** 480 of 2,400 filters, 19.78% of all parameters.
- **Compared with the final unstructured model** (Exercise 4, 70% sparsity, 68.05%), it removes far fewer parameters.
- **But its zeros are structured,** so it is the only model of this lab that could be turned into a **really smaller and faster** network by removing the zero channels.
- **The choice depends on the goal:**
  - fewest non-zero weights → **unstructured 70% (Ex. 4)**;
  - actual speed and memory gains on standard hardware → **structured 20% (Bonus)**.

**Limitations:**
- Single run, validation results only. This model was not part of the test evaluation; it can be checked with `python evaluate.py results/bonus/bonus_structured_20.pt --split test`.
- The same fraction of filters is removed in every layer. A global structured criterion, or a lower ratio for sensitive layers such as the first conv, would probably do better (see Exercises 2 vs 3).
- The physical removal of the channels, and the resulting speed and size gains, were not implemented or measured.

## 6. Conclusion

The bonus is complete:
- a custom structured pruning method was implemented by inheriting from `BasePruningMethod`. It removes the `amount` lowest-L1 filters (output channels) of every weight with more than 2 dimensions, and leaves the other weights untouched;
- it was applied to all conv layers at 0–50%, with and without retraining, and compared with unstructured pruning.

**Main results:**
- **Structured pruning destroys the network without retraining** (chance level from 30% of the filters), much faster than unstructured pruning.
- **With 5 epochs of retraining, it keeps the accuracy within 1 point up to 20% of the filters** (67.76%), and within 3.6 points at 50%.
- It is less accurate than unstructured pruning at the same sparsity. **In exchange, its zeros form whole channels that can be removed to obtain a truly smaller and faster model.**

## 7. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output: zero filters, epochs 0–5, before/after retraining, checks, chosen ratio |
| `bonus_results.csv` | Per ratio: actual sparsity, zero filters, val accuracy/loss before and after, gain, gap |
| `bonus_history.csv` | Every epoch of every model |
| `bonus_curve.png/.pdf` | Structured vs unstructured, before and after retraining, against the actual sparsity |
| `bonus_structured_00.pt` … `bonus_structured_50.pt` | The pruned and retrained models (masks on the conv weights) |
