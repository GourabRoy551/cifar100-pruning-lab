# Exercise 5: Iterative training and pruning

**Code:** `pruning.py`, Exercise 5 block (`run_exercises = [5]`), reusing `fine_tune()` from Exercise 4
**Starting model:** the Exercise 0 baseline (`best_model_params.pt`), loaded **once**
**Evaluation data:** validation set (10,000 images), the same saved split and pre-processing as for training

## 1. Task (from the template)

Instead of doing one pruning step and one retraining step (Exercises 3 + 4), perform them **many times**: prune a little → retrain → prune a little more → retrain → …

Then draw the accuracy / sparsity curve, comment on it, and choose a model. The template's tip is to turn Exercise 4 into a function and iterate over it.

## 2. What was done

| Setting | Value |
|---|---|
| Model | The baseline is loaded **once**, and **the same model object** is pruned and retrained at every step. This is the only exercise where the model is not reloaded, as the lab notes say |
| Pruning steps (total sparsity) | **20% → 40% → 60% → 70% → 80% → 90%** (6 steps) |
| Pruning method at each step | Global L1 unstructured (`global_unstructured` + `L1Unstructured`), all weights and biases, as in Exercise 3 |
| Retraining after each step | `fine_tune()` from Exercise 4: **5 epochs**, SGD lr 0.01 with cosine decay, momentum 0.9, weight decay 5e-4 |
| Model kept at each step | The model after the last retraining epoch, which is also the starting point of the next step |
| Rule for choosing a model | Highest sparsity with validation accuracy at most 1 point below the baseline (≥ 67.67%) |

**How many weights are pruned at each step:**
- Each step removes **only the extra weights needed to reach the new total**: `round(target × 2,820,740) − already pruned`. That is 564,148 weights for the steps to 20, 40 and 60%, and 282,074 for the steps to 70, 80 and 90%.
- The new pruning chooses only among the weights that are still active, using their **current (retrained) values**.
- Weights pruned at earlier steps stay pruned, because the masks accumulate.
- **Check:** the measured sparsity is exactly 20.00, 40.00, 60.00, 70.00, 80.00 and 90.00%. A test on a random model also confirmed that every earlier mask stays included in the later ones.

## 3. Results

Baseline (unpruned): **68.67%** validation accuracy, loss 1.1873.

### Iterative pruning (Exercise 5)

| Step | Total sparsity | Weights pruned in this step | Val. acc. right after pruning | Val. acc. after retraining | Gap to baseline (points) | Val. loss after | Train acc. (last epoch) | Total retraining epochs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20% | 564,148 | 68.61% | 68.52% | 0.15 | 1.2365 | 94.37% | 5 |
| 2 | **40%** | 564,148 | 68.05% | **67.99%** | **0.68** | 1.3007 | 96.10% | 10 |
| 3 | 60% | 564,148 | 65.37% | 67.53% | 1.14 | 1.3423 | 96.51% | 15 |
| 4 | 70% | 282,074 | 63.90% | 67.23% | 1.44 | 1.3850 | 96.56% | 20 |
| 5 | 80% | 282,074 | 52.37% | 66.88% | 1.79 | 1.3975 | 94.09% | 25 |
| 6 | 90% | 282,074 | 17.75% | 66.03% | 2.64 | 1.3059 | 84.56% | 30 |

### Comparison with one-shot pruning + retraining (Exercise 4)

| Sparsity | One-shot, right after pruning (Ex. 3) | Iterative, right after the pruning step | One-shot + retraining (Ex. 4) | Iterative + retraining (Ex. 5) | Iterative − one-shot (points) |
|---:|---:|---:|---:|---:|---:|
| 20% | 68.61% | 68.61% | 68.51% | 68.52% | +0.01 |
| 40% | 68.21% | 68.05% | 68.66% | 67.99% | −0.67 |
| 60% | 65.09% | 65.37% | 68.52% | 67.53% | −0.99 |
| 70% | 58.22% | 63.90% | 68.05% | 67.23% | −0.82 |
| 80% | 34.60% | 52.37% | 67.39% | 66.88% | −0.51 |
| 90% | 4.61% | 17.75% | 65.31% | **66.03%** | **+0.72** |
| *Retraining epochs* | | | *5 per model* | *5 × step number (10–30)* | |

**Figures:**
- `exercise5_curve.png/.pdf`: accuracy and loss vs sparsity, iterative (Ex. 5) vs one-shot + retraining (Ex. 4).
- `exercise5_iterative_history.png/.pdf`: validation accuracy along the whole procedure, showing a drop at each pruning step followed by recovery during retraining.

## 4. Answers

### [QUESTION 1] Draw the accuracy / sparsity ratio curve and comment on the results.

**1. Each pruning step is much less damaging than one-shot pruning, as expected.**
- When the model is pruned gradually, the accuracy right after each pruning step stays much higher than when the same total sparsity is reached in one go (see the "right after pruning" columns):

  | Sparsity | Right after pruning, iterative | Right after pruning, one-shot |
  |---:|---:|---:|
  | 70% | 63.90% | 58.22% |
  | 80% | 52.37% | 34.60% |
  | 90% | 17.75% | 4.61% |

- **Why:** each step removes only a part of the weights, and the model had already **adapted** to the previous pruning.
- **The history figure shows this well:** small drops for the first steps, larger drops at 80% and 90%, and quick recovery during each retraining.

**2. But after retraining, iterative pruning is NOT better than one-shot pruning, except at 90%.**
- **From 40% to 80%,** the iterative models end **0.5 to 1.0 point below** the one-shot retrained models of Exercise 4, for example 67.53% vs 68.52% at 60%, and 67.23% vs 68.05% at 70%.
- **At 20%** they are equal.
- **Only at 90%, the most extreme sparsity, iterative pruning wins:** 66.03% vs 65.31%, +0.72 point.
- **So the general claim that "iterative is better" is only confirmed at very high sparsity in this experiment.**

**3. The main reason is very likely accumulated overfitting from the repeated retraining.**
- The iterative model is retrained **again and again on the same 40,000 training images**: 10 epochs in total at 40%, 20 at 70%, 30 at 90%.
- The training accuracy rises to **about 99.5% during steps 3 and 4** (training loss about 0.08), and the **validation loss keeps increasing**: 1.24 → 1.30 → 1.34 → 1.39 → 1.40 from 20% to 80%.
- For comparison, the one-shot retrained models stay around 1.21–1.25.
- **In other words:** the network memorises the training set more and more, and its validation predictions become more over-confident when wrong. The accuracy drops a little, and the loss more clearly.
- **Restarting at lr 0.01 at every step** also pushes the model away from a well-converged state each time, as already seen in Exercise 4 (the epoch-1 dip).
- **At 90%,** the network has too few weights left to memorise as much: training accuracy is only 84.6%, and the validation loss goes back down to 1.31. Here the gentler, gradual pruning finally pays off.
- **This explanation fits the measured training and validation curves, but it was not tested separately,** for example with fewer epochs per step or a lower learning rate.

**4. Training cost.**
- One iterative run produces all the sparsity levels, but each one is reached after **more training**: the 90% model received 30 retraining epochs, against 5 for the one-shot 90% model.
- So even with **6 times more training**, iterative pruning is only 0.72 point better at 90%, and worse below.
- **With these settings, one-shot pruning + retraining is the more efficient procedure** for this network, except when the target sparsity is extreme.

**5. Other observations:**
- At 20%, the first step is identical to the Exercise 3/4 model (same pruning, same retraining, same seed), and the results agree (68.52% vs 68.51%), a useful consistency check.
- In the last epochs of steps 2–5, the training loss rises a little while the learning rate decreases, e.g. step 3: 0.088 → 0.159, while the validation accuracy improves. A likely reason is that weight decay reduces over-confidence on the training set as the gradient steps get smaller. This was not investigated further and does not change the conclusions.

### [QUESTION 1] Which model would you keep and why?

**Rule result for this exercise: 40% sparsity (iterative)**, with **67.99%** validation accuracy (0.68 points below the baseline). File: `exercise5_iterative_40.pt`.
- The next step, 60%, is at −1.14 points, just outside the 1-point budget.

**Overall, the Exercise 4 model is better than any Exercise 5 model under this rule:**

| Model | Sparsity | Val. accuracy | Gap |
|---|---:|---:|---:|
| Ex. 4, one-shot 70% + retraining | 70% | 68.05% | −0.62 |
| Ex. 5, iterative 40% | 40% | 67.99% | −0.68 |

The Exercise 4 model removes **30 percentage points more weights** for a **higher** accuracy, with less training.

**Final choice across Exercises 2–5: `exercise4_finetuned_70.pt` (70% global sparsity + retraining, 68.05%).**

**If a larger accuracy loss were acceptable** (about 3 points), the **iterative 90% model** (66.03%, only 10% of the weights left) would be the best very-high-compression model, better than one-shot at 90% (65.31%). This is the regime where iterative pruning is useful here.

**How iterative pruning could do better (not tested):**
- fewer epochs or a lower learning rate per step, to limit the accumulated overfitting;
- more, smaller pruning steps;
- stronger regularisation or data augmentation during retraining;
- keeping the best-validation epoch of each step. This would require a separate data split to avoid choosing on the validation set used for reporting.

**Limitations:**
- These are single-run results on the validation set. Differences below about 0.3 points are within noise.
- Iterative and one-shot models received different total training budgets.
- The final chosen model still has to be evaluated **once on the test set**.

## 5. Conclusion

Exercise 5 is complete:
- the same model was pruned and retrained **6 times** (20 → 90% global sparsity), reusing the Exercise 4 retraining function;
- the masks accumulated correctly and the exact target sparsity was reached at each step;
- the accuracy / sparsity curve was drawn, compared with one-shot pruning, and a model was chosen.

**Main results:**
- **Iterative pruning makes each pruning step much less damaging** (e.g. 52% vs 35% right after pruning to 80%).
- **After retraining it does not beat one-shot pruning + retraining, except at 90%** (+0.72 point), and it ends 0.5–1.0 point lower between 40% and 80%.
- The most likely reason is accumulated overfitting from repeated retraining on the same data (training accuracy up to 99.5%, validation loss increasing).

**Kept model:**
- for this exercise: **iterative 40%** (67.99%);
- overall: **Exercise 4's 70% one-shot + retrained model** (68.05%) remains the best choice under the 1-point rule.

## 6. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output: each step (weights pruned, epochs 0–5, before/after, sparsity check), chosen model |
| `exercise5_results.csv` | One row per step: sparsity, weights pruned, val acc/loss after pruning and after retraining, total epochs |
| `exercise5_history.csv` | Every epoch of every step |
| `exercise5_curve.png/.pdf` | Accuracy and loss vs sparsity: iterative vs one-shot + retraining |
| `exercise5_iterative_history.png/.pdf` | Validation accuracy along the whole iterative procedure |
| `exercise5_iterative_20.pt` … `exercise5_iterative_90.pt` | The model after each step (with masks) |
