# Final evaluation on the test set

**Code:** `evaluate.py`, an independent evaluation script, as recommended in the lab instructions. It loads saved models only: no training, no pruning, no exercise result is overwritten.
**Data:**
- **validation set:** 10,000 images from the saved split `cifar100_split.pt`;
- **test set:** the official CIFAR-100 test set, 10,000 images, never used before this step.

## 1. Purpose

During Exercises 0–5, **all model choices were made on the validation set** (best epoch of the baseline, kept ratio of each exercise, final choice).

The test set was kept aside until the choices were fixed. It is used here **once**, to report how the chosen models perform on images that played no part in any decision. This avoids an optimistic result caused by selecting models on the data used to report them.

**Models evaluated, all chosen beforehand on validation:**

| Model | Why it is evaluated |
|---|---|
| Baseline (Ex. 0) | Reference (unpruned) |
| Global 50%, no retraining (Ex. 3) | Model kept in Exercise 3 |
| **Global 70% + retraining (Ex. 4)** | **Final choice**: highest sparsity within 1 point of the baseline on validation, across Exercises 2–5 |
| Global 80% + retraining (Ex. 4) | Candidate that narrowly missed the 1-point rule on validation (−1.28) |
| Iterative 40% (Ex. 5) | Model kept in Exercise 5 |
| Iterative 90% (Ex. 5) | Best very-high-compression model (66.03% on validation) |

## 2. Results

| Model | Sparsity | Non-zero parameters | Val. accuracy | **Test accuracy** | Test − val (points) | Test gap to baseline (points) | Test loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Ex. 0) | 0% | 2,820,740 | 68.67% | **68.54%** | −0.13 | — | 1.1898 |
| Global 50%, no retraining (Ex. 3) | 50% | 1,410,370 | 67.67% | **67.37%** | −0.30 | −1.17 | 1.2215 |
| **Global 70% + retraining (Ex. 4), final choice** | **70%** | **846,222** | 68.05% | **68.34%** | +0.29 | **−0.20** | 1.2081 |
| Global 80% + retraining (Ex. 4) | 80% | 564,148 | 67.39% | **67.84%** | +0.45 | −0.70 | 1.2001 |
| Iterative 40% (Ex. 5) | 40% | 1,692,444 | 67.99% | **67.76%** | −0.23 | −0.78 | 1.2943 |
| Iterative 90% (Ex. 5) | 90% | 282,074 | 66.03% | **66.09%** | +0.06 | −2.45 | 1.3132 |

**Figure:** `test_accuracy.png/.pdf` shows the test accuracy of each model, with the baseline as a dashed line and the final choice in orange.

The sparsity values measured on the loaded models match the targets exactly, which confirms that the saved files are the expected pruned models.

## 3. Analysis

### Validation and test agree

- **Small differences:** for every model, test accuracy is within **±0.5 point** of validation accuracy (between −0.30 and +0.45).
- **No sign of overfitting to the validation set:** the choices made on validation hold on new data.
- **Statistical precision:** with 10,000 test images and an accuracy near 68%, the statistical uncertainty of one accuracy value is about ±0.47 point (one binomial standard error, √(0.68 × 0.32 / 10,000)). Validation–test differences of this size are expected.
- **Same ordering:** the conclusions from the validation results are confirmed on the test set.
  - **Retraining matters.** The retrained 70% model (68.34%) beats the unretrained 50% model (67.37%), with 20 points more sparsity.
  - **High sparsity has a cost.** The 90% model loses about 2.5 points.

### The final model

**Global 70% pruning + retraining (Exercise 4), `results/exercise4/exercise4_finetuned_70.pt`:**
- **Test accuracy: 68.34%**, against **68.54%** for the unpruned baseline. That is a loss of only **0.20 point** (20 images out of 10,000), smaller than the statistical uncertainty.
- **It uses only 30% of the weights:** 846,222 non-zero parameters instead of 2,820,740.
- **Conclusion:** **70% of the network's weights can be removed with practically no loss of test accuracy**, using global L1 pruning followed by 5 epochs of retraining.

### About the other models

- **Global 80% + retraining scores 67.84% on test,** which is −0.70 point and so *within* 1 point of the baseline, although it missed the rule on validation (−1.28).
  - **The final choice is not changed:** switching to the 80% model because of its test score would mean selecting on the test set, which is exactly what this protocol avoids.
  - It can be reported as a sign that **80% sparsity is close to the limit** of what this network tolerates with 5 epochs of retraining.
- **Iterative 40% (67.76%)** is below the one-shot 70% model on test as well, with 30 points less sparsity. This confirms the Exercise 5 conclusion: with these settings, iterative pruning did not beat one-shot pruning + retraining at moderate sparsity.
- **Iterative 90% (66.09%)** keeps about 96% of the baseline accuracy (66.09 / 68.54) with only **10% of the weights** (282,074). It is the best option if very high compression is needed and a loss of about 2.5 points is acceptable.

### Reminder: sparsity is not file size or speed

- **File size:** all these models are stored as dense tensors. The pruned files are as large as, or larger than, the baseline (masked files store `weight_orig` + `weight_mask`; see Exercise 1).
- **Speed:** inference is not faster on standard hardware.
- **What would be needed:** turning the 70% sparsity into real memory or speed gains needs sparse storage or sparse computation, or **structured** pruning that removes whole channels (Bonus exercise).

## 4. Conclusion

- **Baseline:** **68.54%** test accuracy (68.67% validation), within the 65–70% target of the lab.
- **Final pruned model:** global 70% sparsity + retraining, **68.34%** test accuracy. This is **0.20 point** below the baseline, with **70% fewer non-zero weights**.
- **Validation and test results agree within ±0.5 point** for all models, so the model selection done on validation is reliable.

## 5. Files in this folder

| File | Content |
|---|---|
| `log.txt` | Full output: each model's file, sparsity, validation and test loss/accuracy |
| `test_results.csv` | The same results in a table (including the number of correct test predictions) |
| `test_accuracy.png/.pdf` | Test accuracy of each evaluated model |
