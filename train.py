"""
Note: a pre-configured virtual environment is available on the CREMI computer here:
/net/ens/DeepLearning/python/tensorflow2
"""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # only save plots to files, never open a window
import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from src.model import ConvNet

"""
Exercise 0: train a convolution network on Cifar100.
Implement the whole training loop (and the base model) for the image classification task on Cifar100.


TIPS:
1) Fulfil the following TODOs if you have unsure how to do it.
2) Don't use a too complex and too large network as baseline because the pruning procedure will be harder to implement. I advice to limit
    yourself to few layers (conv, bn, relu, maxpool, linear) and to a "small" architecture. Try to aim accuracy al least around 75%.
    Basically, MobileNet-like or ResNet18 architecture should be enough.

[VERY IMPORTANT] save the train-val split so you can use the same split whe re-training the network.
    Otherwise you could have data leakage leading to overestimate your model performances.


TIPS:
1) Simple data augmentation is enough
2) Save the model and backup the file in case of you override the original model. You will have to retrain all models otherwise...
3) Learning rate scheduling is good to improve performances, but should  be necessary
4) if you're struggling to get good enough accuracy, switch the dataset to Cifar10 (performances should be around 90% easily)
"""


# Hyper-parameters
n_epoch = 30
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# All paths are relative to this file, so the script can be launched from anywhere.
base_dir = Path(__file__).resolve().parent
best_model_params_path = str(base_dir / 'best_model_params.pt')
results_dir = base_dir / "results" / "exercise0"
results_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed)

# Everything passed to log() is printed and also written to results/exercise0/train_log.txt.
log_file = open(results_dir / "train_log.txt", "w", encoding="utf-8")


def log(message=""):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


# TODO - Step 1: Define Pre-processing or input data and target labels:
#   - Use the data augmentation classes from 'torchvision.transforms' package. The minimal augmentation required is 'ToTensor'
#   - Consider adding normalization (between -1 and +1), and other data augmentation is the performances are a bit low.

input_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
target_transform = None
eval_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# TODO - Step 2: Load the data
#   - Load Cifar100 dataset from torchvision
#   - Retrieve both train and validation sets. We will use the validation set for our last model evaluation (test set)
#       and split the training set into two subsets (80%-20%): 1 for training, and a second one for validation.

data_dir = base_dir / "data"
train_dataset = CIFAR100(
    root=data_dir, train=True, download=True,
    transform=input_transform, target_transform=target_transform,
)
validation_dataset = CIFAR100(
    root=data_dir, train=True, download=True, transform=eval_transform,
)
test_dataset = CIFAR100(
    root=data_dir, train=False, download=True, transform=eval_transform,
)

# The split is saved once and reloaded afterwards, so every run uses the same train/val images.
split_path = base_dir / "cifar100_split.pt"
try:
    split = torch.load(split_path, map_location="cpu", weights_only=True)
except FileNotFoundError:
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    n_train = int(0.8 * len(train_dataset))
    split = {
        "train_indices": indices[:n_train],
        "val_indices": indices[n_train:],
    }
    torch.save(split, split_path)

train_indices = split["train_indices"]
val_indices = split["val_indices"]
assert len(train_indices) == int(0.8 * len(train_dataset))
assert set(train_indices).isdisjoint(val_indices)
assert sorted(train_indices + val_indices) == list(range(len(train_dataset)))

train_set = torch.utils.data.Subset(train_dataset, train_indices)
val_set = torch.utils.data.Subset(validation_dataset, val_indices)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
log(f"Train / validation / test samples: {len(train_set)} / {len(val_set)} / {len(test_dataset)}")

# TODO - Step 3: Implement the model
#   - In "model.py" file, implement the BasicConvNet class
device = "cpu" if device == "gpu" else device
model = ConvNet().to(device=device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Model: ResNet18-like ConvNet, {n_params:,} trainable parameters")
log(f"Epochs: {n_epoch}, batch size: {batch_size}, lr: {lr}, momentum: {momentum}, weight decay: {weight_decay}")
log()


# TODO - Step 4: Training configuration
#   - Define the loss function (cf "CrossEntropyLoss" class)
#   - Define the optimiser (SGD with learning rate of 10e-2)
#   - Define the metrics (Accuracy). You can use the package 'torchmetrics' if available.

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
train_accuracy = MulticlassAccuracy(num_classes=100, average="micro").to(device)
val_accuracy = MulticlassAccuracy(num_classes=100, average="micro").to(device)

# TODO - Step 5: Implement the training loop and the validation loop
#   - Fill the missing TODOs in the following training loop
best_val_acc = -1.
best_epoch = 0
history = []  # one row per epoch, saved to results/exercise0/history.csv
start_time = time.time()

for epoch in range(1, n_epoch + 1):
    log(f'Epoch {epoch:02d}/{n_epoch:02d}')
    log('-' * 10)
    epoch_start = time.time()
    current_lr = optimizer.param_groups[0]["lr"]

    # TODO - 5.1: Training loop
    #   - Implement each of the following steps
    model.train()
    train_ma_loss = 0.
    train_ma_acc = 0.
    train_loss_sum = 0.
    train_count = 0
    train_accuracy.reset()
    train_bar = tqdm(train_loader, desc="Train", unit="batch", leave=False)
    for step, (inp, out) in enumerate(train_bar):
        # TODO: put data to the correct device
        inp = inp.to(device)
        out = out.to(device)

        # TODO: Reset gradients stored in the optimiser
        optimizer.zero_grad(set_to_none=True)

        # TODO: Perform forward pass
        predictions = model(inp)

        # TODO: Compute loss function and accuracy between predictions and ground truth
        loss = loss_fn(predictions, out)
        train_accuracy.update(predictions.detach(), out)

        # TODO: Do backpropagation
        loss.backward()

        # TODO: Update optimiser internal parameters and metrics
        optimizer.step()

        # TODO: Update moving average loss and accuracy
        train_loss_sum += loss.item() * out.size(0)
        train_count += out.size(0)
        train_ma_loss = train_loss_sum / train_count
        if (step + 1) % 20 == 0 or step + 1 == len(train_loader):
            train_ma_acc = train_accuracy.compute().item()
            train_bar.set_postfix(loss=f"{train_ma_loss:.4f}", acc=f"{train_ma_acc:.4f}")

    val_loss = 0.
    val_acc = 0.
    model.eval()
    # TODO - 5.2: Validation loop
    val_loss_sum = 0.
    val_count = 0
    val_accuracy.reset()
    with torch.no_grad():
        for step, (inp, out) in enumerate(tqdm(val_loader, desc="Val", unit="batch", leave=False)):
            # TODO: put data to the correct device
            inp = inp.to(device)
            out = out.to(device)

            # TODO: Perform forward pass
            predictions = model(inp)

            # TODO: Compute loss function between predictions and ground truth
            loss = loss_fn(predictions, out)

            # TODO: Update moving average loss and accuracy
            val_loss_sum += loss.item() * out.size(0)
            val_count += out.size(0)
            val_accuracy.update(predictions, out)
            val_loss = val_loss_sum / val_count
    val_acc = val_accuracy.compute().item()

    epoch_time = time.time() - epoch_start
    log(f"lr = {current_lr:.5f} | train loss = {train_ma_loss:.4f}, train acc = {train_ma_acc:.4f} | "
        f"val loss = {val_loss:.4f}, val acc = {val_acc:.4f} | {epoch_time:.1f} s")

    if best_val_acc < val_acc:
        torch.save(model.state_dict(), best_model_params_path)
        log(f"Saving model with better perfs ({best_val_acc:.4f} -> {val_acc:.4f})")
        best_val_acc = val_acc
        best_epoch = epoch
    scheduler.step()

    # Save the history after every epoch, so it is kept even if training is interrupted.
    history.append({
        "epoch": epoch, "lr": current_lr,
        "train_loss": train_ma_loss, "train_acc": train_ma_acc,
        "val_loss": val_loss, "val_acc": val_acc,
        "epoch_time_s": round(epoch_time, 2),
    })
    with open(results_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    log()

total_time = time.time() - start_time

# Final evaluation: reload the best checkpoint (selected on validation) and evaluate it on the test set.
test_model = ConvNet().to(device)
test_model.load_state_dict(
    torch.load(best_model_params_path, map_location=device, weights_only=True)
)
test_model.eval()
test_accuracy = MulticlassAccuracy(num_classes=100, average="micro").to(device)
test_loss_sum = 0.
test_count = 0
with torch.no_grad():
    for inp, out in tqdm(test_loader, desc="Test", unit="batch"):
        inp = inp.to(device)
        out = out.to(device)
        predictions = test_model(inp)
        loss = loss_fn(predictions, out)
        test_loss_sum += loss.item() * out.size(0)
        test_count += out.size(0)
        test_accuracy.update(predictions, out)
test_loss = test_loss_sum / test_count
test_acc = test_accuracy.compute().item()

best_row = history[best_epoch - 1]
log("=" * 40)
log(f"Training time: {total_time / 60:.1f} min")
log(f"Best epoch: {best_epoch}")
log(f"Best validation accuracy: {best_val_acc:.2%} (val loss {best_row['val_loss']:.4f})")
log(f"Test loss: {test_loss:.4f} - test accuracy: {test_acc:.2%}")
torch.save(test_model.state_dict(), best_model_params_path + ".backup")

# Summary of the run for the report.
summary = {
    "model": "ResNet18-like ConvNet (src/model.py)",
    "trainable_parameters": n_params,
    "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
    "epochs": n_epoch, "batch_size": batch_size, "lr": lr, "momentum": momentum,
    "weight_decay": weight_decay, "scheduler": "CosineAnnealingLR", "seed": seed,
    "train_samples": len(train_set), "val_samples": len(val_set), "test_samples": len(test_dataset),
    "best_epoch": best_epoch,
    "best_val_acc": best_val_acc, "best_val_loss": best_row["val_loss"],
    "train_acc_at_best_epoch": best_row["train_acc"],
    "test_acc": test_acc, "test_loss": test_loss,
    "training_time_min": round(total_time / 60, 2),
}
with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Plots for the report (saved as PNG and PDF in results/exercise0/).
train_color, val_color, test_color = "#2a78d6", "#eb6834", "#555555"
epochs = [row["epoch"] for row in history]

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
ax_loss.plot(epochs, [row["train_loss"] for row in history], color=train_color, linewidth=2, label="Train")
ax_loss.plot(epochs, [row["val_loss"] for row in history], color=val_color, linewidth=2, label="Validation")
ax_loss.set_title("Cross-entropy loss")
ax_loss.set_ylabel("Loss")

ax_acc.plot(epochs, [100 * row["train_acc"] for row in history], color=train_color, linewidth=2, label="Train")
ax_acc.plot(epochs, [100 * row["val_acc"] for row in history], color=val_color, linewidth=2, label="Validation")
ax_acc.axhline(100 * test_acc, color=test_color, linewidth=1.5, linestyle="--", label=f"Test, best model ({test_acc:.2%})")
ax_acc.plot(best_epoch, 100 * best_val_acc, "o", color=val_color, markersize=8, markeredgecolor="white")
ax_acc.annotate(f"best val {best_val_acc:.2%}\n(epoch {best_epoch})", (best_epoch, 100 * best_val_acc),
                textcoords="offset points", xytext=(-10, -35), ha="right", fontsize=9)
ax_acc.set_title("Accuracy")
ax_acc.set_ylabel("Accuracy (%)")

for ax in (ax_loss, ax_acc):
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(results_dir / "training_curves.png", dpi=200)
fig.savefig(results_dir / "training_curves.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(epochs, [row["lr"] for row in history], color=train_color, linewidth=2)
ax.set_title("Learning rate (cosine annealing)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning rate")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(results_dir / "learning_rate.png", dpi=200)
fig.savefig(results_dir / "learning_rate.pdf")
plt.close(fig)

log(f"Results saved in {results_dir}")
log_file.close()
