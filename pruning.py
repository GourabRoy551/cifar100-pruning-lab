import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only save plots to files, never open a window
import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from src.model import ConvNet

# Choose which exercise(s) to run, e.g. [1], then [2], ... or ["bonus"]. Each one saves its own results in results/.
run_exercises = ["bonus"]

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
seed = 42
# Sparsity ratios tested in the accuracy / sparsity curves (finer steps at low sparsity, where accuracy changes fastest).
sparsity_ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Rule used to choose a model: the highest sparsity whose validation accuracy is at most 1 point below the baseline.
accuracy_tolerance = 0.01
# Re-training of the pruned models (Exercises 4 and 5): a few epochs, learning rate 10x lower than for the baseline (0.1).
finetune_epochs = 5
finetune_lr = 0.01
# Exercise 5: total sparsity reached after each iterative pruning step (re-training after every step).
iterative_ratios = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
# Bonus: fraction of filters (output channels) removed in every conv layer by the structured pruning method.
bonus_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
base_dir = Path(__file__).resolve().parent
filename = base_dir / "best_model_params.pt"  # trained baseline from Exercise 0 (never overwritten here)
results_dir = base_dir / "results"

# Validation data: same pre-processing and same saved train/val split as in train.py.
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
val_dataset = CIFAR100(root=base_dir / "data", train=True, download=True, transform=eval_transform)
split = torch.load(base_dir / "cifar100_split.pt", map_location="cpu", weights_only=True)
val_loader = DataLoader(Subset(val_dataset, split["val_indices"]), batch_size=batch_size, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss()

# Training data (only used to re-train pruned models, Exercises 4 and 5): same split and augmentation as in train.py.
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = CIFAR100(root=base_dir / "data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(Subset(train_dataset, split["train_indices"]), batch_size=batch_size, shuffle=True)


def load_model():
    """Build a new model and load the original (unpruned) trained weights."""
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate(model, loader):
    """Return (mean loss, accuracy) of the model on a data loader."""
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    with torch.no_grad():
        for inp, out in loader:
            inp, out = inp.to(device), out.to(device)
            predictions = model(inp)
            loss_sum += loss_fn(predictions, out).item() * out.size(0)
            correct += (predictions.argmax(dim=1) == out).sum().item()
            count += out.size(0)
    return loss_sum / count, correct / count


def parameters_to_prune(model):
    """List of (layer, name) for every weight and bias of every layer (conv, batch norm and linear)."""
    params = []
    for layer in model.modules():
        for name in ("weight", "bias"):
            if isinstance(getattr(layer, name, None), torch.Tensor):
                params.append((layer, name))
    return params


def effective_value(layer, name):
    """Weights actually used by the layer: weight_orig * weight_mask if the tensor is pruned, else the tensor itself."""
    if hasattr(layer, name + "_mask"):
        return getattr(layer, name + "_orig") * getattr(layer, name + "_mask")
    return getattr(layer, name)


def layer_sparsity(model):
    """Number of parameters and of zeros in each layer (weights and biases counted together)."""
    rows = []
    for layer_name, layer in model.named_modules():
        values = [effective_value(layer, name) for name in ("weight", "bias")
                  if isinstance(getattr(layer, name, None), torch.Tensor)]
        if values:
            n_params = sum(value.numel() for value in values)
            n_zeros = sum((value == 0).sum().item() for value in values)
            rows.append({"layer": layer_name, "type": type(layer).__name__,
                         "parameters": n_params, "zeros": n_zeros, "sparsity": n_zeros / n_params})
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def choose_ratio(rows, baseline_acc):
    """Row with the highest sparsity whose validation accuracy is within 'accuracy_tolerance' of the baseline."""
    acceptable = [row for row in rows if row["val_acc"] >= baseline_acc - accuracy_tolerance]
    return max(acceptable, key=lambda row: row["sparsity_ratio"], default=rows[0])


def plot_accuracy_curves(curves, baseline_acc, title, path, kept_row=None,
                         colors=("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
    """Validation accuracy (left) and loss (right) vs sparsity ratio. 'curves' is a list of (label, rows)."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    for (label, rows), color in zip(curves, colors):
        x = [100 * row["sparsity_ratio"] for row in rows]
        ax_acc.plot(x, [100 * row["val_acc"] for row in rows], "-o", color=color, linewidth=2, markersize=6, label=label)
        ax_loss.plot(x, [row["val_loss"] for row in rows], "-o", color=color, linewidth=2, markersize=6, label=label)
    ax_acc.axhline(100 * baseline_acc, color="#555555", linestyle="--", linewidth=1.5,
                   label=f"Unpruned baseline ({baseline_acc:.2%})")
    ax_acc.axhline(1, color="#999999", linestyle=":", linewidth=1.5, label="Chance level (1%)")
    if kept_row is not None:
        ax_acc.annotate(f"kept: {kept_row['sparsity_ratio']:.0%} sparsity\n{kept_row['val_acc']:.2%} val acc",
                        (100 * kept_row["sparsity_ratio"], 100 * kept_row["val_acc"]),
                        textcoords="offset points", xytext=(15, 10), fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "#555555"})
    ax_acc.set_title("Validation accuracy")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_loss.set_title("Validation loss")
    ax_loss.set_ylabel("Cross-entropy loss")
    for ax in (ax_acc, ax_loss):
        ax.set_xlabel("Sparsity ratio (%)")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    # One shared legend below the two plots, so it never hides a curve.
    handles, labels = ax_acc.get_legend_handles_labels()
    n_columns = len(labels) if len(labels) <= 4 else 3  # wrap on two rows when there are many curves
    fig.legend(handles, labels, loc="lower center", ncol=n_columns, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07 if len(labels) <= 4 else 0.13, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_layer_sparsity(layer_rows, sparsity_ratio, title, path):
    """Horizontal bar chart of the sparsity of each layer, colored by layer type."""
    type_colors = {"Conv2d": "#2a78d6", "BatchNorm2d": "#eb6834", "Linear": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(8, 10))
    y = range(len(layer_rows))
    ax.barh(y, [100 * row["sparsity"] for row in layer_rows], height=0.7,
            color=[type_colors[row["type"]] for row in layer_rows])
    ax.axvline(100 * sparsity_ratio, color="#555555", linestyle="--", linewidth=1.5)
    ax.set_yticks(list(y), [row["layer"] for row in layer_rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Sparsity of the layer (%)")
    ax.set_title(title)
    for layer_type, color in type_colors.items():
        ax.barh([0], [0], color=color, label=layer_type)  # empty bars, only for the legend
    ax.plot([], [], color="#555555", linestyle="--", label=f"Target ({sparsity_ratio:.0%})")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# Everything passed to log() is printed and also written to the log file of the current exercise.
log_file = None


def log(message=""):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


"""
Exercise 1: Pruning a pre-trained model

Given a pre-trained model, we will perform model pruning with the following properties:
- granularity: unconstrained pruning
- layer-by-layer pruning
- weight selection: L1 norm

PyTorch provides few utility functions to prune models, especially the most common one. Especially in our case, the
function is already implemented:
>> torch.nn.utils.prune.l1_unstructured
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html

To use l1_unstructured, we need to provide the layer with the weights to prune, but also the weight's name. After loading
the pre-trained model, here are the steps that you should follow to perform pruning:

1) retrieve a list of all layers
2) for each layer, get the different weights
3) prune the weights using l1_unstructured (bias included) for a given sparsity ratio
"""

if 1 in run_exercises:
    out_dir = results_dir / "exercise1"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Exercise 1: layer-by-layer L1 unstructured pruning")
    log("=" * 50)

    # TODO: load the-pretrained model
    model = load_model()
    baseline_loss, baseline_acc = evaluate(model, val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")

    sparsity_ratio = 0.2

    # 1) retrieve a list of all layers, 2) for each layer, get the different weights (weight and bias)
    params = parameters_to_prune(model)
    n_layers = len({id(layer) for layer, _ in params})
    log(f"{n_layers} layers with parameters, {len(params)} tensors to prune (weights + biases)")

    # Copy of one layer's weights before pruning, to show in a plot which weights L1 pruning removes.
    example_layer = "layer3.0.conv2"
    weights_before = model.get_submodule(example_layer).weight.detach().cpu().flatten().clone()

    # TODO: implement the pruning loop:
    #   tips: you can check which indexes of the original weights have been retained by printing 'model.named_buffers()'
    # 3) In every tensor, the 20% entries with the smallest absolute value (L1 norm) are set to 0.
    for layer, name in params:
        prune.l1_unstructured(layer, name=name, amount=sparsity_ratio)

    # Each pruned tensor 'weight' is now 'weight_orig' (original values) * 'weight_mask' (buffer of 0/1).
    masks = [(buffer_name, buffer) for buffer_name, buffer in model.named_buffers() if buffer_name.endswith("_mask")]
    log(f"{len(masks)} masks created by pruning, for example:")
    for buffer_name, mask in masks[:5]:
        log(f"  {buffer_name}: shape {tuple(mask.shape)}, kept {mask.float().mean().item():.2%} of the entries")
    log("  Mask of the first 3x3 kernel of the first conv (1 = kept, 0 = pruned):")
    for mask_row in model.stem[0].weight_mask[0, 0].int().tolist():
        log(f"    {mask_row}")

    pruned_loss, pruned_acc = evaluate(model, val_loader)
    layer_rows = layer_sparsity(model)
    total_params = sum(row["parameters"] for row in layer_rows)
    total_zeros = sum(row["zeros"] for row in layer_rows)
    log(f"Pruned model: {total_zeros:,} / {total_params:,} parameters are zero "
        f"(actual sparsity = {total_zeros / total_params:.2%})")
    log(f"Pruned model: val loss = {pruned_loss:.4f}, val accuracy = {pruned_acc:.2%} "
        f"(drop of {100 * (baseline_acc - pruned_acc):.2f} points)")

    # Checkpoint with the pruning re-parametrization (stores weight_orig AND weight_mask).
    masked_path = out_dir / "exercise1_pruned_20_masked.pt"
    torch.save(model.state_dict(), masked_path)

    # TODO: compress the model weight file.
    #   Pruning using the prune package are performed on the original model, but the original weights are not removed. Basically,
    #   it simply copies the original weights, and apply the pruning on these weights. To permanently remove the original weights,
    #   you need to call on each layer the function 'prune.remove' that takes as input the layer and the weight to remove.
    for layer, name in params:
        prune.remove(layer, name)
    compressed_path = out_dir / "exercise1_pruned_20.pt"
    torch.save(model.state_dict(), compressed_path)

    # After prune.remove the zeros are kept, so the accuracy must be exactly the same.
    removed_loss, removed_acc = evaluate(model, val_loader)
    log(f"After prune.remove: val accuracy = {removed_acc:.2%} (same model, masks removed)")

    size_baseline = filename.stat().st_size / 1e6
    size_masked = masked_path.stat().st_size / 1e6
    size_compressed = compressed_path.stat().st_size / 1e6
    log(f"File sizes: baseline = {size_baseline:.2f} MB, pruned with masks = {size_masked:.2f} MB, "
        f"pruned after prune.remove = {size_compressed:.2f} MB")

    # Results for the report.
    save_csv([{
        "sparsity_ratio": sparsity_ratio, "actual_sparsity": total_zeros / total_params,
        "total_parameters": total_params, "zero_parameters": total_zeros,
        "baseline_val_loss": baseline_loss, "baseline_val_acc": baseline_acc,
        "pruned_val_loss": pruned_loss, "pruned_val_acc": pruned_acc,
        "after_remove_val_acc": removed_acc, "acc_drop_points": 100 * (baseline_acc - pruned_acc),
        "size_baseline_mb": round(size_baseline, 2), "size_masked_mb": round(size_masked, 2),
        "size_after_remove_mb": round(size_compressed, 2),
    }], out_dir / "exercise1_results.csv")
    save_csv(layer_rows, out_dir / "exercise1_layer_sparsity.csv")

    # Plot 1: sparsity of each layer (layer-by-layer pruning gives ~20% in every layer).
    plot_layer_sparsity(layer_rows, sparsity_ratio,
                        f"Exercise 1: sparsity per layer (layer-by-layer pruning, {sparsity_ratio:.0%})",
                        out_dir / "exercise1_layer_sparsity")

    # Plot 2: weight distribution of one layer; the shaded band around 0 is what L1 pruning removed.
    weights_after = model.get_submodule(example_layer).weight.detach().cpu().flatten()
    threshold = weights_before[weights_after == 0].abs().max().item()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts, _, _ = ax.hist(weights_before.numpy(), bins=120, color="#2a78d6")
    ax.set_ylim(0, 1.3 * counts.max())  # free space at the top for the legend
    ax.axvspan(-threshold, threshold, ymax=1 / 1.3, color="#eb6834", alpha=0.25,
               label=f"Pruned: |w| < {threshold:.4f} ({sparsity_ratio:.0%} smallest)")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Number of weights")
    ax.set_title(f"Exercise 1: weights of {example_layer} before pruning")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "exercise1_weight_histogram.png", dpi=200)
    fig.savefig(out_dir / "exercise1_weight_histogram.pdf")
    plt.close(fig)

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None


"""
Exercise 2: Accuracy / sparsity ratio curve.

In order to find the optimal compression ratio, we need to evaluate our pruned model on the validation set. So, for
different sparsity ratio, prune the network, and evaluate the pruned model on Cifar100 validation set.

<<Warning>> be sure that the random train/val split is the same split you used for training the original model!

Tips:
- move Exercise 1 code into a function, and call that function with different sparsity ratio.
- Remember that the original model is lost, so you have to reload the original weights before doing another pruning procedure.
- Save the tuple (sparsity ratio , validation accuracy) for each model (and maybe the pruned weights)  to avoid a long retraining procedure in case the script fails.

[QUESTION] Comments the obtained curve. which sparsity ratio would you keep and why?
"""


def layerwise_pruning(model, sparsity_ratio):
    """Exercise 1 as a function: prune the weights and biases of each layer with l1_unstructured (layer by layer)."""
    for layer, name in parameters_to_prune(model):
        prune.l1_unstructured(layer, name=name, amount=sparsity_ratio)


if 2 in run_exercises:
    out_dir = results_dir / "exercise2"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Exercise 2: accuracy / sparsity ratio curve (layer-by-layer L1 pruning)")
    log("=" * 50)

    baseline_loss, baseline_acc = evaluate(load_model(), val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")
    log(f"Sparsity ratios: {sparsity_ratios}")
    log()

    results = []
    for sparsity_ratio in sparsity_ratios:
        # The original model is lost after pruning, so a new model with the original weights is loaded for every ratio.
        model = load_model()
        layerwise_pruning(model, sparsity_ratio)
        val_loss, val_acc = evaluate(model, val_loader)
        layer_rows = layer_sparsity(model)
        actual_sparsity = sum(row["zeros"] for row in layer_rows) / sum(row["parameters"] for row in layer_rows)

        # Save the pruned weights (after prune.remove, so the file only holds the final weights with zeros).
        for layer, name in parameters_to_prune(model):
            prune.remove(layer, name)
        checkpoint = f"exercise2_layerwise_{round(100 * sparsity_ratio):02d}.pt"
        torch.save(model.state_dict(), out_dir / checkpoint)

        # Save the (sparsity ratio, validation accuracy) results after every ratio, in case the script fails.
        results.append({
            "sparsity_ratio": sparsity_ratio, "actual_sparsity": actual_sparsity,
            "val_loss": val_loss, "val_acc": val_acc,
            "acc_drop_points": 100 * (baseline_acc - val_acc), "checkpoint": checkpoint,
        })
        save_csv(results, out_dir / "exercise2_results.csv")
        log(f"sparsity {sparsity_ratio:4.0%} (actual {actual_sparsity:.2%}): val loss = {val_loss:.4f}, "
            f"val accuracy = {val_acc:.2%} (drop of {100 * (baseline_acc - val_acc):.2f} points)")

    kept = choose_ratio(results, baseline_acc)
    log()
    log(f"Kept ratio (highest sparsity with accuracy at most {100 * accuracy_tolerance:.0f} point below baseline): "
        f"{kept['sparsity_ratio']:.0%}, val accuracy = {kept['val_acc']:.2%}")

    plot_accuracy_curves([("Layer-by-layer L1 pruning", results)], baseline_acc,
                         "Exercise 2: accuracy / sparsity ratio curve (layer-by-layer pruning, no retraining)",
                         out_dir / "exercise2_curve", kept_row=kept)

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None




"""
Exercise 3: Global pruning

Instead of considering each layer individually, we can prune the whole network by removing the lowest weights in the entire model.
This is achieved by calling the function 'global_unstructured'.
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html

This function takes three arguments:
1) :parameter: a list of tuples that contains the layer and the weight's name (similar to those given in 'l1_unstructured'
2) :pruning_method: the function to be called for pruning. Here, we will use 'prune.L1Unstructured'
3) :amount: the sparsity ratio.

[QUESTION] Similarly to the exercise 2, draw the accuracy / sparsity ratio curve and comments the obtained results. For the best
model, what is the sparsity ratio of each layer?

Tips: Save the intermediate models to accelerate the next exercise.
"""


def global_pruning(model, sparsity_ratio):
    """Prune the 'sparsity_ratio' smallest weights and biases (L1) of the whole network at once."""
    prune.global_unstructured(parameters_to_prune(model), pruning_method=prune.L1Unstructured, amount=sparsity_ratio)


if 3 in run_exercises:
    out_dir = results_dir / "exercise3"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Exercise 3: global L1 unstructured pruning")
    log("=" * 50)

    baseline_loss, baseline_acc = evaluate(load_model(), val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")
    log(f"Sparsity ratios: {sparsity_ratios}")
    log()

    results = []
    layer_rows_per_ratio = {}
    for sparsity_ratio in sparsity_ratios:
        # As in Exercise 2, a new model with the original weights is loaded for every ratio.
        model = load_model()
        global_pruning(model, sparsity_ratio)
        val_loss, val_acc = evaluate(model, val_loader)
        layer_rows = layer_sparsity(model)
        layer_rows_per_ratio[sparsity_ratio] = layer_rows
        actual_sparsity = sum(row["zeros"] for row in layer_rows) / sum(row["parameters"] for row in layer_rows)

        # Global threshold: every weight / bias of the network with |value| <= threshold was set to 0.
        pruned_values = torch.cat([getattr(layer, name + "_orig")[getattr(layer, name + "_mask") == 0].abs()
                                   for layer, name in parameters_to_prune(model)])
        threshold = pruned_values.max().item() if pruned_values.numel() > 0 else 0.

        # The masks are kept in the saved model (weight_orig + weight_mask), so that Exercise 4 can re-train
        # the pruned model while the pruned weights stay at zero.
        checkpoint = f"exercise3_global_{round(100 * sparsity_ratio):02d}.pt"
        torch.save(model.state_dict(), out_dir / checkpoint)

        results.append({
            "sparsity_ratio": sparsity_ratio, "actual_sparsity": actual_sparsity, "threshold": threshold,
            "val_loss": val_loss, "val_acc": val_acc,
            "acc_drop_points": 100 * (baseline_acc - val_acc), "checkpoint": checkpoint,
        })
        save_csv(results, out_dir / "exercise3_results.csv")
        log(f"sparsity {sparsity_ratio:4.0%} (actual {actual_sparsity:.2%}, |w| <= {threshold:.4f} pruned): "
            f"val loss = {val_loss:.4f}, val accuracy = {val_acc:.2%} (drop of {100 * (baseline_acc - val_acc):.2f} points)")

    kept = choose_ratio(results, baseline_acc)
    log()
    log(f"Kept ratio (highest sparsity with accuracy at most {100 * accuracy_tolerance:.0f} point below baseline): "
        f"{kept['sparsity_ratio']:.0%}, val accuracy = {kept['val_acc']:.2%}")

    # Sparsity of each layer for the kept (best) model.
    kept_layers = layer_rows_per_ratio[kept["sparsity_ratio"]]
    save_csv(kept_layers, out_dir / "exercise3_layer_sparsity.csv")
    log()
    log(f"Sparsity of each layer for the kept model ({kept['sparsity_ratio']:.0%}):")
    for row in kept_layers:
        log(f"  {row['layer']:22s} {row['type']:12s} {row['parameters']:8,d} params  sparsity = {row['sparsity']:.2%}")
    log("Sparsity per layer type:")
    for layer_type in ("Conv2d", "BatchNorm2d", "Linear"):
        rows = [row for row in kept_layers if row["type"] == layer_type]
        n_params = sum(row["parameters"] for row in rows)
        n_zeros = sum(row["zeros"] for row in rows)
        log(f"  {layer_type:12s} {n_zeros:,} / {n_params:,} zeros, sparsity = {n_zeros / n_params:.2%}")

    # Curve, compared with the layer-by-layer results of Exercise 2 when they exist.
    curves = []
    exercise2_csv = results_dir / "exercise2" / "exercise2_results.csv"
    if exercise2_csv.exists():
        with open(exercise2_csv) as f:
            layerwise_results = [{"sparsity_ratio": float(row["sparsity_ratio"]), "val_acc": float(row["val_acc"]),
                                  "val_loss": float(row["val_loss"])} for row in csv.DictReader(f)]
        curves.append(("Layer-by-layer L1 pruning (Ex. 2)", layerwise_results))
    curves.append(("Global L1 pruning", results))
    plot_accuracy_curves(curves, baseline_acc,
                         "Exercise 3: accuracy / sparsity ratio curve (global vs layer-by-layer pruning, no retraining)",
                         out_dir / "exercise3_curve", kept_row=kept)
    plot_layer_sparsity(kept_layers, kept["sparsity_ratio"],
                        f"Exercise 3: sparsity per layer of the kept model (global pruning, {kept['sparsity_ratio']:.0%})",
                        out_dir / "exercise3_layer_sparsity")

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None




"""
Exercise 4: Re-training a pruned model

For each model saved in Exercise 3, re-train the network on the training set for a few epoch with a lower learning rate.

[QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?
"""


def load_pruned_model(path):
    """Build a new model and load a pruned model saved with its masks (weight_orig + weight_mask)."""
    model = ConvNet().to(device)
    # Add the pruning re-parametrization first, so that the model has the same keys as the saved file.
    for layer, name in parameters_to_prune(model):
        prune.identity(layer, name)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def fine_tune(model, n_epochs, lr, desc=""):
    """Re-train a pruned model on the training set. The masks stay active, so the pruned weights remain zero
    (only 'weight_orig' is trained, and 'weight = weight_orig * weight_mask').
    Returns one row per epoch, epoch 0 being the model before re-training."""
    torch.manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    val_loss, val_acc = evaluate(model, val_loader)
    history = [{"epoch": 0, "lr": "", "train_loss": "", "train_acc": "", "val_loss": val_loss, "val_acc": val_acc}]
    log(f"  epoch 0 (before re-training): val loss = {val_loss:.4f}, val accuracy = {val_acc:.2%}")

    for epoch in range(1, n_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum, correct, count = 0., 0, 0
        for inp, out in tqdm(train_loader, desc=f"{desc} epoch {epoch}/{n_epochs}", unit="batch", leave=False):
            inp, out = inp.to(device), out.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inp)
            loss = loss_fn(predictions, out)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * out.size(0)
            correct += (predictions.argmax(dim=1) == out).sum().item()
            count += out.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader)
        history.append({"epoch": epoch, "lr": current_lr, "train_loss": loss_sum / count, "train_acc": correct / count,
                        "val_loss": val_loss, "val_acc": val_acc})
        log(f"  epoch {epoch}/{n_epochs}: lr = {current_lr:.5f} | train loss = {loss_sum / count:.4f}, "
            f"train acc = {correct / count:.4f} | val loss = {val_loss:.4f}, val acc = {val_acc:.4f}")
    model.eval()
    return history


if 4 in run_exercises:
    out_dir = results_dir / "exercise4"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Exercise 4: re-training the pruned models of Exercise 3")
    log("=" * 50)

    baseline_loss, baseline_acc = evaluate(load_model(), val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")
    log(f"Re-training: {finetune_epochs} epochs, SGD lr = {finetune_lr} (cosine decay), momentum 0.9, weight decay 5e-4")
    log()

    results = []
    all_history = []
    for sparsity_ratio in sparsity_ratios:
        name = f"{round(100 * sparsity_ratio):02d}"
        log(f"Sparsity {sparsity_ratio:.0%}: re-training exercise3_global_{name}.pt")
        model = load_pruned_model(results_dir / "exercise3" / f"exercise3_global_{name}.pt")
        history = fine_tune(model, finetune_epochs, finetune_lr, desc=f"sparsity {sparsity_ratio:.0%}")

        # Check that the masks stayed active: the sparsity must be the same as before re-training.
        layer_rows = layer_sparsity(model)
        actual_sparsity = sum(row["zeros"] for row in layer_rows) / sum(row["parameters"] for row in layer_rows)

        # Saved with the masks, like in Exercise 3.
        checkpoint = f"exercise4_finetuned_{name}.pt"
        torch.save(model.state_dict(), out_dir / checkpoint)

        for row in history:
            all_history.append({"sparsity_ratio": sparsity_ratio, **row})
        save_csv(all_history, out_dir / "exercise4_history.csv")

        before, after = history[0], history[-1]
        results.append({
            "sparsity_ratio": sparsity_ratio, "actual_sparsity": actual_sparsity,
            "val_loss_before": before["val_loss"], "val_acc_before": before["val_acc"],
            "val_loss": after["val_loss"], "val_acc": after["val_acc"],
            "gain_points": 100 * (after["val_acc"] - before["val_acc"]),
            "acc_drop_points": 100 * (baseline_acc - after["val_acc"]), "checkpoint": checkpoint,
        })
        save_csv(results, out_dir / "exercise4_results.csv")
        log(f"  -> val accuracy {before['val_acc']:.2%} before, {after['val_acc']:.2%} after re-training "
            f"(gain of {100 * (after['val_acc'] - before['val_acc']):.2f} points, "
            f"{100 * (baseline_acc - after['val_acc']):.2f} points below baseline), actual sparsity = {actual_sparsity:.2%}")
        log()

    kept = choose_ratio(results, baseline_acc)
    log(f"Kept ratio (highest sparsity with accuracy at most {100 * accuracy_tolerance:.0f} point below baseline): "
        f"{kept['sparsity_ratio']:.0%}, val accuracy = {kept['val_acc']:.2%}")

    # Curve: global pruning before (Exercise 3) and after re-training.
    before_rows = [{"sparsity_ratio": row["sparsity_ratio"], "val_acc": row["val_acc_before"],
                    "val_loss": row["val_loss_before"]} for row in results]
    plot_accuracy_curves([("Global pruning, before re-training (Ex. 3)", before_rows),
                          ("Global pruning + re-training (Ex. 4)", results)], baseline_acc,
                         f"Exercise 4: accuracy / sparsity ratio curve (re-training {finetune_epochs} epochs, lr {finetune_lr})",
                         out_dir / "exercise4_curve", kept_row=kept, colors=("#eb6834", "#1baf7a"))

    # Validation accuracy during re-training, for the high sparsity ratios (where re-training matters most).
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # light = less sparse, dark = more sparse
    shown_ratios = [ratio for ratio in sparsity_ratios if ratio >= 0.5]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ratio, color in zip(shown_ratios, ramp):
        rows = [row for row in all_history if row["sparsity_ratio"] == ratio]
        ax.plot([row["epoch"] for row in rows], [100 * row["val_acc"] for row in rows], "-o",
                color=color, linewidth=2, markersize=6, label=f"{ratio:.0%} sparsity")
    ax.axhline(100 * baseline_acc, color="#555555", linestyle="--", linewidth=1.5,
               label=f"Unpruned baseline ({baseline_acc:.2%})")
    ax.set_xticks(range(finetune_epochs + 1))
    ax.set_xlabel("Re-training epoch (0 = right after pruning)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exercise 4: validation accuracy during re-training")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "exercise4_retraining_history.png", dpi=200)
    fig.savefig(out_dir / "exercise4_retraining_history.pdf")
    plt.close(fig)

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None




"""
Exercise 5: Iterative training and pruning

Instead of doing 1 pruning and fine-tuning step, you can perform it many times.

[QUESTION 1] Similarly, draw the accuracy / sparsity ratio curve and comments the obtained results. Which model would you keep and why ?

[TIPS]
1) move exercise 4 into a function and iterate over it.
"""

if 5 in run_exercises:
    out_dir = results_dir / "exercise5"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Exercise 5: iterative pruning and re-training")
    log("=" * 50)

    # The baseline is loaded only ONCE: the same model is pruned and re-trained again and again.
    model = load_model()
    baseline_loss, baseline_acc = evaluate(model, val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")
    log(f"Sparsity steps: {iterative_ratios}")
    log(f"After each step: re-training {finetune_epochs} epochs, SGD lr = {finetune_lr} (cosine decay)")
    log()

    params = parameters_to_prune(model)
    n_total = sum(getattr(layer, name).numel() for layer, name in params)
    results = []
    all_history = []
    for step, sparsity_ratio in enumerate(iterative_ratios, start=1):
        # Number of weights to prune at this step so that the TOTAL sparsity reaches 'sparsity_ratio'.
        # The weights pruned at the previous steps stay pruned, and the new pruning only chooses among the remaining ones.
        n_already_pruned = sum((getattr(layer, name + "_mask") == 0).sum().item()
                               for layer, name in params if hasattr(layer, name + "_mask"))
        n_to_prune = round(sparsity_ratio * n_total) - n_already_pruned
        log(f"Step {step}: pruning {n_to_prune:,} more weights (global L1) to reach {sparsity_ratio:.0%} total sparsity")
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=n_to_prune,
                                  importance_scores={(layer, name): effective_value(layer, name).detach()
                                                     for layer, name in params})

        # Exercise 4 as a function: re-train the pruned model (epoch 0 = right after this pruning step).
        history = fine_tune(model, finetune_epochs, finetune_lr, desc=f"step {step} ({sparsity_ratio:.0%})")

        layer_rows = layer_sparsity(model)
        actual_sparsity = sum(row["zeros"] for row in layer_rows) / sum(row["parameters"] for row in layer_rows)

        # Saved with the masks; the same model object is then pruned further at the next step.
        checkpoint = f"exercise5_iterative_{round(100 * sparsity_ratio):02d}.pt"
        torch.save(model.state_dict(), out_dir / checkpoint)

        for row in history:
            all_history.append({"step": step, "sparsity_ratio": sparsity_ratio, **row})
        save_csv(all_history, out_dir / "exercise5_history.csv")

        before, after = history[0], history[-1]
        results.append({
            "step": step, "sparsity_ratio": sparsity_ratio, "actual_sparsity": actual_sparsity,
            "pruned_this_step": n_to_prune,
            "val_loss_after_pruning": before["val_loss"], "val_acc_after_pruning": before["val_acc"],
            "val_loss": after["val_loss"], "val_acc": after["val_acc"],
            "gain_points": 100 * (after["val_acc"] - before["val_acc"]),
            "acc_drop_points": 100 * (baseline_acc - after["val_acc"]),
            "total_retraining_epochs": step * finetune_epochs, "checkpoint": checkpoint,
        })
        save_csv(results, out_dir / "exercise5_results.csv")
        log(f"  -> val accuracy {before['val_acc']:.2%} right after pruning, {after['val_acc']:.2%} after re-training "
            f"({100 * (baseline_acc - after['val_acc']):.2f} points below baseline), actual sparsity = {actual_sparsity:.2%}, "
            f"{step * finetune_epochs} re-training epochs in total")
        log()

    # The unpruned baseline is the starting point (0%) of the curve.
    curve_rows = [{"sparsity_ratio": 0.0, "val_acc": baseline_acc, "val_loss": baseline_loss}] + results
    kept = choose_ratio(curve_rows, baseline_acc)
    log(f"Kept ratio (highest sparsity with accuracy at most {100 * accuracy_tolerance:.0f} point below baseline): "
        f"{kept['sparsity_ratio']:.0%}, val accuracy = {kept['val_acc']:.2%}")

    # Curve, compared with one-shot pruning + re-training (Exercise 4) when its results exist.
    curves = []
    colors = []
    exercise4_csv = results_dir / "exercise4" / "exercise4_results.csv"
    if exercise4_csv.exists():
        with open(exercise4_csv) as f:
            oneshot_results = [{"sparsity_ratio": float(row["sparsity_ratio"]), "val_acc": float(row["val_acc"]),
                                "val_loss": float(row["val_loss"])} for row in csv.DictReader(f)]
        curves.append(("One-shot global pruning + re-training (Ex. 4)", oneshot_results))
        colors.append("#1baf7a")
    curves.append(("Iterative pruning + re-training (Ex. 5)", curve_rows))
    colors.append("#eda100")
    plot_accuracy_curves(curves, baseline_acc,
                         f"Exercise 5: accuracy / sparsity ratio curve (iterative vs one-shot, {finetune_epochs} epochs per re-training)",
                         out_dir / "exercise5_curve", kept_row=kept, colors=colors)

    # Validation accuracy along the whole iterative procedure: drops at each pruning step, recovery during re-training.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x, y = [0], [100 * baseline_acc]
    for step, sparsity_ratio in enumerate(iterative_ratios, start=1):
        start = (step - 1) * finetune_epochs
        for row in all_history:
            if row["step"] == step:
                x.append(start + row["epoch"])
                y.append(100 * row["val_acc"])
        ax.axvline(start, color="#999999", linestyle=":", linewidth=1)
        ax.text(start + finetune_epochs / 2, 97, f"{sparsity_ratio:.0%}", ha="center", va="top", fontsize=9)
    ax.plot(x, y, "-o", color="#2a78d6", linewidth=2, markersize=5, label="Iterative pruning + re-training")
    ax.axhline(100 * baseline_acc, color="#555555", linestyle="--", linewidth=1.5,
               label=f"Unpruned baseline ({baseline_acc:.2%})")
    ax.set_xlabel("Total re-training epochs (dotted lines = pruning steps, top labels = total sparsity)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exercise 5: validation accuracy during iterative pruning and re-training")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "exercise5_iterative_history.png", dpi=200)
    fig.savefig(out_dir / "exercise5_iterative_history.pdf")
    plt.close(fig)

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None




"""
Bonus Exercise: Custom weight selection function

Instead of using the provided function for the weight selection you can also implement your own by inheriting from the
'BasePruningMethod' class.
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html#torch.nn.utils.prune.BasePruningMethod


Implement the structured granularity to remove the 'amount' lowest columns (using L1 norm) for the weight with dimensionality > 2.
"""

class MyPruningMethod(prune.BasePruningMethod):
    """Structured pruning: for a weight with more than 2 dimensions (conv weight: out x in x k x k), remove the
    'amount' fraction of filters (output channels, dim 0) with the lowest L1 norm. Other weights are not pruned.
    """
    PRUNING_TYPE = 'structured'
    dim = 0  # pruned dimension: whole filters / output channels

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, t, default_mask):
        """

        :param t: default tensor (weights)
        :param default_mask: mask obtained from a previous pruning procedure.
        :return: a mask with values in {0, 1} with the same dims as t.
        """
        mask = default_mask.clone()  # always clone to avoid memory corruption
        if t.dim() <= 2:
            return mask  # only weights with dimensionality > 2 are pruned

        # L1 norm of each filter: sum of |w| over its input channels and kernel positions.
        filter_norms = (t * default_mask).abs().sum(dim=tuple(range(1, t.dim())))
        n_pruned = round(self.amount * t.shape[0])
        if n_pruned > 0:
            pruned_filters = torch.topk(filter_norms, k=n_pruned, largest=False).indices
            mask[pruned_filters] = 0  # the whole filter is set to zero
        return mask


def structured_pruning(model, amount):
    """Apply MyPruningMethod to the weight of every conv layer (layer by layer, same fraction of filters in each)."""
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            MyPruningMethod.apply(layer, "weight", amount)


def count_zero_filters(model):
    """Number of filters (output channels) whose weights are all zero, and total number of filters, in the conv layers."""
    n_zero, n_total = 0, 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            weight = effective_value(layer, "weight")
            n_zero += (weight.abs().sum(dim=(1, 2, 3)) == 0).sum().item()
            n_total += weight.shape[0]
    return n_zero, n_total


if "bonus" in run_exercises:
    out_dir = results_dir / "bonus"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")
    log("Bonus: structured pruning (L1 norm of whole filters) with a custom BasePruningMethod")
    log("=" * 50)

    baseline_loss, baseline_acc = evaluate(load_model(), val_loader)
    log(f"Baseline (unpruned): val loss = {baseline_loss:.4f}, val accuracy = {baseline_acc:.2%}")
    log(f"Fraction of filters removed in every conv layer: {bonus_ratios}")
    log(f"Re-training after pruning: {finetune_epochs} epochs, SGD lr = {finetune_lr} (cosine decay)")
    log()

    results = []
    all_history = []
    for filter_ratio in bonus_ratios:
        # A new model with the original weights for every ratio (independent experiments, as in Exercises 2-4).
        model = load_model()
        structured_pruning(model, filter_ratio)
        n_zero_filters, n_filters = count_zero_filters(model)
        log(f"Filter ratio {filter_ratio:.0%}: {n_zero_filters} / {n_filters} filters set to zero")

        history = fine_tune(model, finetune_epochs, finetune_lr, desc=f"filters {filter_ratio:.0%}")

        # Check that the pruned filters are still entirely zero after re-training.
        n_zero_after, _ = count_zero_filters(model)
        layer_rows = layer_sparsity(model)
        actual_sparsity = sum(row["zeros"] for row in layer_rows) / sum(row["parameters"] for row in layer_rows)

        checkpoint = f"bonus_structured_{round(100 * filter_ratio):02d}.pt"
        torch.save(model.state_dict(), out_dir / checkpoint)

        for row in history:
            all_history.append({"filter_ratio": filter_ratio, **row})
        save_csv(all_history, out_dir / "bonus_history.csv")

        before, after = history[0], history[-1]
        results.append({
            "sparsity_ratio": filter_ratio, "actual_sparsity": actual_sparsity,
            "zero_filters": n_zero_after, "total_filters": n_filters,
            "val_loss_before": before["val_loss"], "val_acc_before": before["val_acc"],
            "val_loss": after["val_loss"], "val_acc": after["val_acc"],
            "gain_points": 100 * (after["val_acc"] - before["val_acc"]),
            "acc_drop_points": 100 * (baseline_acc - after["val_acc"]), "checkpoint": checkpoint,
        })
        save_csv(results, out_dir / "bonus_results.csv")
        log(f"  -> val accuracy {before['val_acc']:.2%} right after pruning, {after['val_acc']:.2%} after re-training "
            f"({100 * (baseline_acc - after['val_acc']):.2f} points below baseline), "
            f"zero filters after re-training = {n_zero_after} / {n_filters}, actual sparsity = {actual_sparsity:.2%}")
        log()

    kept = choose_ratio(results, baseline_acc)
    log(f"Kept ratio (highest filter ratio with accuracy at most {100 * accuracy_tolerance:.0f} point below baseline): "
        f"{kept['sparsity_ratio']:.0%} of the filters, val accuracy = {kept['val_acc']:.2%}")

    # Curve against the actual sparsity, compared with unstructured global pruning (Exercises 3 and 4) when available.
    def read_results(path, acc_key, loss_key):
        with open(path) as f:
            return [{"sparsity_ratio": float(row["sparsity_ratio"]), "val_acc": float(row[acc_key]),
                     "val_loss": float(row[loss_key])} for row in csv.DictReader(f)
                    if float(row["sparsity_ratio"]) <= max(bonus_ratios) + 0.1]

    curves, colors = [], []
    exercise3_csv = results_dir / "exercise3" / "exercise3_results.csv"
    exercise4_csv = results_dir / "exercise4" / "exercise4_results.csv"
    if exercise3_csv.exists():
        curves.append(("Unstructured global, no re-training (Ex. 3)", read_results(exercise3_csv, "val_acc", "val_loss")))
        colors.append("#eb6834")
    if exercise4_csv.exists():
        curves.append(("Unstructured global + re-training (Ex. 4)", read_results(exercise4_csv, "val_acc", "val_loss")))
        colors.append("#1baf7a")
    curves.append(("Structured (filters), no re-training", [
        {"sparsity_ratio": row["actual_sparsity"], "val_acc": row["val_acc_before"], "val_loss": row["val_loss_before"]}
        for row in results]))
    colors.append("#eda100")
    curves.append(("Structured (filters) + re-training", [
        {"sparsity_ratio": row["actual_sparsity"], "val_acc": row["val_acc"], "val_loss": row["val_loss"]}
        for row in results]))
    colors.append("#e87ba4")
    kept_point = {"sparsity_ratio": kept["actual_sparsity"], "val_acc": kept["val_acc"]}
    plot_accuracy_curves(curves, baseline_acc,
                         "Bonus: structured (whole filters) vs unstructured pruning, against the actual sparsity",
                         out_dir / "bonus_curve", kept_row=kept_point, colors=colors)

    log(f"Results saved in {out_dir}")
    log_file.close()
    log_file = None
