"""
Prepares the data files used by the PGFPlots figures of the report (report/pgf_figures/).
It only READS the existing results (results/...) and the baseline model; nothing there is modified.
Accuracies are converted to percentages so the figures can plot them directly.

Run from the lab directory:  python report/data/prepare_figure_data.py
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

base_dir = Path(__file__).resolve().parents[2]
results = base_dir / "results"
out_dir = Path(__file__).resolve().parent


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(name, header, rows):
    with open(out_dir / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def pct(value):
    return round(100 * float(value), 2)


# Exercise 0: training history.
rows = read(results / "exercise0" / "history.csv")
write("ex0_history.csv", ["epoch", "train_acc", "val_acc", "train_loss", "val_loss"],
      [[r["epoch"], pct(r["train_acc"]), pct(r["val_acc"]), round(float(r["train_loss"]), 4),
        round(float(r["val_loss"]), 4)] for r in rows])
summary = json.load(open(results / "exercise0" / "summary.json"))

# Exercise 1: weight histogram of layer3.0.conv2 (all weights, and the weights removed by 20% L1 pruning).
layer = "layer3.0.conv2.weight"
before = torch.load(base_dir / "best_model_params.pt", map_location="cpu", weights_only=True)[layer].flatten()
after = torch.load(results / "exercise1" / "exercise1_pruned_20.pt", map_location="cpu", weights_only=True)[layer].flatten()
pruned = before[after == 0]
threshold = pruned.abs().max().item()
edges = np.linspace(-0.12, 0.12, 101)
all_counts, _ = np.histogram(before.clamp(-0.12, 0.12).numpy(), bins=edges)
pruned_counts, _ = np.histogram(pruned.numpy(), bins=edges)
hist_rows = [[round(edges[i], 5), int(all_counts[i]), int(pruned_counts[i])] for i in range(len(all_counts))]
hist_rows.append([round(edges[-1], 5), 0, 0])  # closing edge for 'ybar interval'
write("ex1_histogram.csv", ["left_edge", "all", "pruned"], hist_rows)

# Exercises 2-5: accuracy / loss vs sparsity.
ex2 = read(results / "exercise2" / "exercise2_results.csv")
write("ex2.csv", ["sparsity", "acc", "loss"],
      [[pct(r["sparsity_ratio"]), pct(r["val_acc"]), round(float(r["val_loss"]), 4)] for r in ex2])
ex3 = read(results / "exercise3" / "exercise3_results.csv")
write("ex3.csv", ["sparsity", "acc", "loss"],
      [[pct(r["sparsity_ratio"]), pct(r["val_acc"]), round(float(r["val_loss"]), 4)] for r in ex3])
ex4 = read(results / "exercise4" / "exercise4_results.csv")
write("ex4.csv", ["sparsity", "before", "after", "gain", "loss_after"],
      [[pct(r["sparsity_ratio"]), pct(r["val_acc_before"]), pct(r["val_acc"]), round(float(r["gain_points"]), 2),
        round(float(r["val_loss"]), 4)] for r in ex4])
baseline_acc = pct(ex2[0]["val_acc"])
baseline_loss = round(float(ex2[0]["val_loss"]), 4)
ex5 = read(results / "exercise5" / "exercise5_results.csv")
write("ex5.csv", ["sparsity", "after_pruning", "after_retraining", "loss_after"],
      [[0.0, baseline_acc, baseline_acc, baseline_loss]] +
      [[pct(r["sparsity_ratio"]), pct(r["val_acc_after_pruning"]), pct(r["val_acc"]), round(float(r["val_loss"]), 4)]
       for r in ex5])

# Exercise 5: timeline of the whole iterative procedure (x = total re-training epochs).
history = read(results / "exercise5" / "exercise5_history.csv")
epochs_per_step = max(int(r["epoch"]) for r in history)
timeline = [[0, baseline_acc]] + [[(int(r["step"]) - 1) * epochs_per_step + int(r["epoch"]), pct(r["val_acc"])]
                                  for r in history]
write("ex5_timeline.csv", ["x", "acc"], timeline)

# Exercise 3: sparsity per stage / layer type of the kept 50% model.
layers = read(results / "exercise3" / "exercise3_layer_sparsity.csv")
groups = [("Stem conv", lambda r: r["layer"] == "stem.0"),
          ("Stage 1", lambda r: r["layer"].startswith("layer1") and r["type"] == "Conv2d"),
          ("Stage 2", lambda r: r["layer"].startswith("layer2") and r["type"] == "Conv2d"),
          ("Stage 3", lambda r: r["layer"].startswith("layer3") and r["type"] == "Conv2d"),
          ("Stage 4", lambda r: r["layer"].startswith("layer4") and r["type"] == "Conv2d"),
          ("Classifier", lambda r: r["type"] == "Linear"),
          ("All BN", lambda r: r["type"] == "BatchNorm2d")]
stage_rows = []
for label, keep in groups:
    selected = [r for r in layers if keep(r)]
    n_params = sum(int(r["parameters"]) for r in selected)
    n_zeros = sum(int(r["zeros"]) for r in selected)
    stage_rows.append([label, n_params, round(100 * n_zeros / n_params, 2)])
write("ex3_stages.csv", ["part", "parameters", "sparsity"], stage_rows)

# Final evaluation (validation and test accuracy of the selected models).
final = read(results / "final_evaluation" / "test_results.csv")
write("final.csv", ["index", "sparsity", "val", "test"],
      [[i, pct(r["sparsity"]), pct(r["val_acc"]), pct(r["test_acc"])] for i, r in enumerate(final)])

# Bonus: structured pruning (x = actual sparsity).
bonus = read(results / "bonus" / "bonus_results.csv")
write("bonus.csv", ["sparsity", "before", "after"],
      [[pct(r["actual_sparsity"]), pct(r["val_acc_before"]), pct(r["val_acc"])] for r in bonus])

print(f"Baseline val acc = {baseline_acc}, test acc = {pct(summary['test_acc'])}, Ex.1 threshold = {threshold:.4f}")
print(f"Files written to {out_dir}")
