"""
Independent evaluation script: evaluates saved models without re-training or re-pruning anything.

Usage:
    python evaluate.py                                   -> evaluates the models listed in 'models_to_evaluate'
                                                            on the validation AND test sets, saves results + plot
    python evaluate.py path/to/model.pt [--split test]   -> evaluates a single saved model (val or test, default: val)

Works with plain models (e.g. best_model_params.pt) and with pruned models saved with their masks
(weight_orig + weight_mask, e.g. Exercises 3-5).
"""
import argparse
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

from src.model import ConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"

# Models evaluated by default. They were all chosen beforehand on the VALIDATION set (Exercises 0-5);
# the test set is only used here, once, to report their final performance.
models_to_evaluate = [
    ("Baseline (Ex. 0)", base_dir / "best_model_params.pt"),
    ("Global 50%, no re-training (Ex. 3)", results_dir / "exercise3" / "exercise3_global_50.pt"),
    ("Global 70% + re-training (Ex. 4) - FINAL CHOICE", results_dir / "exercise4" / "exercise4_finetuned_70.pt"),
    ("Global 80% + re-training (Ex. 4)", results_dir / "exercise4" / "exercise4_finetuned_80.pt"),
    ("Iterative 40% (Ex. 5)", results_dir / "exercise5" / "exercise5_iterative_40.pt"),
    ("Iterative 90% (Ex. 5)", results_dir / "exercise5" / "exercise5_iterative_90.pt"),
]

# Same pre-processing (no augmentation) and same saved train/val split as in train.py.
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
split = torch.load(base_dir / "cifar100_split.pt", map_location="cpu", weights_only=True)
val_dataset = CIFAR100(root=base_dir / "data", train=True, download=True, transform=eval_transform)
test_dataset = CIFAR100(root=base_dir / "data", train=False, download=True, transform=eval_transform)
loaders = {
    "val": DataLoader(Subset(val_dataset, split["val_indices"]), batch_size=batch_size, shuffle=False),
    "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
}
loss_fn = torch.nn.CrossEntropyLoss()


def load_any_model(path):
    """Load a plain model or a pruned model saved with its masks."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = ConvNet().to(device)
    # For pruned models, add the pruning re-parametrization first so the keys match (weight_orig + weight_mask).
    for layer_name, layer in model.named_modules():
        for name in ("weight", "bias"):
            if f"{layer_name}.{name}_mask" in state:
                prune.identity(layer, name)
    model.load_state_dict(state)
    model.eval()
    return model


def sparsity(model):
    """Fraction of weights and biases that are exactly zero (all conv, batch norm and linear layers)."""
    n_zeros, n_total = 0, 0
    for layer in model.modules():
        for name in ("weight", "bias"):
            if isinstance(getattr(layer, name, None), torch.Tensor):
                value = (getattr(layer, name + "_orig") * getattr(layer, name + "_mask")
                         if hasattr(layer, name + "_mask") else getattr(layer, name))
                n_zeros += (value == 0).sum().item()
                n_total += value.numel()
    return n_zeros / n_total, n_total - n_zeros


def evaluate(model, loader):
    """Return (mean loss, accuracy, number of correct predictions, number of images)."""
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    with torch.no_grad():
        for inp, out in loader:
            inp, out = inp.to(device), out.to(device)
            predictions = model(inp)
            loss_sum += loss_fn(predictions, out).item() * out.size(0)
            correct += (predictions.argmax(dim=1) == out).sum().item()
            count += out.size(0)
    return loss_sum / count, correct / count, correct, count


parser = argparse.ArgumentParser(description="Evaluate saved models on the validation / test sets.")
parser.add_argument("checkpoint", nargs="?", help="a single model file to evaluate (optional)")
parser.add_argument("--split", choices=["val", "test"], default="val", help="data used for a single model")
args = parser.parse_args()

if args.checkpoint:
    # Single model: print the result only (nothing is saved or overwritten).
    model = load_any_model(args.checkpoint)
    model_sparsity, n_nonzero = sparsity(model)
    loss, acc, correct, count = evaluate(model, loaders[args.split])
    print(f"{args.checkpoint}")
    print(f"  sparsity = {model_sparsity:.2%} ({n_nonzero:,} non-zero parameters)")
    print(f"  {args.split}: loss = {loss:.4f}, accuracy = {acc:.2%} ({correct}/{count})")
else:
    out_dir = results_dir / "final_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")

    def log(message=""):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    log("Final evaluation of the chosen models (validation and test sets)")
    log("=" * 50)
    rows = []
    for label, path in models_to_evaluate:
        if not path.exists():
            log(f"{label}: file not found ({path}), skipped")
            continue
        model = load_any_model(path)
        model_sparsity, n_nonzero = sparsity(model)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"])
        test_loss, test_acc, test_correct, test_count = evaluate(model, loaders["test"])
        rows.append({"model": label, "file": str(path.relative_to(base_dir)), "sparsity": model_sparsity,
                     "nonzero_parameters": n_nonzero, "val_loss": val_loss, "val_acc": val_acc,
                     "test_loss": test_loss, "test_acc": test_acc, "test_correct": test_correct,
                     "test_images": test_count})
        log(f"{label}")
        log(f"  file: {path.relative_to(base_dir)}")
        log(f"  sparsity = {model_sparsity:.2%} ({n_nonzero:,} non-zero parameters)")
        log(f"  val:  loss = {val_loss:.4f}, accuracy = {val_acc:.2%}")
        log(f"  test: loss = {test_loss:.4f}, accuracy = {test_acc:.2%} ({test_correct}/{test_count})")

    with open(out_dir / "test_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Plot: test accuracy of each model (the final choice in orange), with its sparsity.
    # A dot plot with a zoomed axis, because the differences between models are only a few points.
    baseline_test_acc = rows[0]["test_acc"]
    accuracies = [100 * row["test_acc"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.8))
    y = range(len(rows))
    colors = ["#eb6834" if "FINAL" in row["model"] else "#2a78d6" for row in rows]
    ax.scatter(accuracies, y, s=80, color=colors, zorder=3)
    for i, row in enumerate(rows):
        ax.text(100 * row["test_acc"] + 0.15, i, f"{row['test_acc']:.2%}  (sparsity {row['sparsity']:.0%})",
                va="center", fontsize=9, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.axvline(100 * baseline_test_acc, color="#555555", linestyle="--", linewidth=1.5,
               label=f"Baseline test accuracy ({baseline_test_acc:.2%})")
    ax.set_yticks(list(y), [row["model"].replace(" - FINAL CHOICE", "\n(final choice)") for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(accuracies) - 1, max(accuracies) + 2.5)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Final evaluation on the CIFAR-100 test set (10,000 images)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "test_accuracy.png", dpi=200)
    fig.savefig(out_dir / "test_accuracy.pdf")
    plt.close(fig)

    log(f"Results saved in {out_dir}")
    log_file.close()
