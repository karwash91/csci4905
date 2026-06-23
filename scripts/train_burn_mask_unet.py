"""Train a small U-Net to predict the burned mask from the initial grid."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import deque
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

LABEL_TO_IDX = {"subcritical": 0, "critical": 1, "supercritical": 2}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}


def reachable_burn_mask(grid: np.ndarray) -> np.ndarray:
    """Flood-fill occupied cells reachable from initial burning cells."""
    occupied = grid != 0
    burning = grid == 2
    visited = np.zeros_like(occupied, dtype=bool)
    queue = deque((int(row), int(col)) for row, col in np.argwhere(burning))
    rows, cols = occupied.shape

    for row, col in queue:
        visited[row, col] = True

    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and occupied[next_row, next_col]
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

    return visited.astype(np.float32)


def label_from_fraction(values: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(values), dtype=np.int64)
    labels[(values >= 0.10) & (values < 0.90)] = LABEL_TO_IDX["critical"]
    labels[values >= 0.90] = LABEL_TO_IDX["supercritical"]
    return labels


class BurnMaskDataset(Dataset):
    """Two-channel input: occupied cells and ignition cells."""

    def __init__(self, data_dir: Path):
        self.grids = np.load(data_dir / "grids.npy")
        self.rows = self._load_metadata(data_dir / "metadata.csv")
        self.labels = np.array([LABEL_TO_IDX[row["label"]] for row in self.rows])
        self.masks = np.stack([reachable_burn_mask(grid) for grid in self.grids])

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int):
        grid = self.grids[index]
        occupied = (grid != 0).astype(np.float32)
        burning = (grid == 2).astype(np.float32)
        inputs = torch.tensor(np.stack([occupied, burning]), dtype=torch.float32)
        mask = torch.tensor(self.masks[index], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return inputs, mask, label


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BurnMaskUNet(nn.Module):
    """Small segmentation model for approximating the fire reachability mask."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(2, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec2 = self.dec2(torch.cat([self.up2(bottleneck), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.output(dec1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/spatial_64"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/burn_mask_unet")
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=4905)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split(
    labels: np.ndarray, seed: int
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    train_indices, val_indices, test_indices = [], [], []

    for label_idx in sorted(set(labels.tolist())):
        indices = np.where(labels == label_idx)[0]
        rng.shuffle(indices)
        n_train = int(0.70 * len(indices))
        n_val = int(0.15 * len(indices))
        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train : n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val :].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def make_loaders(dataset: BurnMaskDataset, batch_size: int, seed: int):
    train_idx, val_idx, test_idx = stratified_split(dataset.labels, seed)
    loaders = {
        "train": DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        ),
        "val": DataLoader(Subset(dataset, val_idx), batch_size=batch_size),
        "test": DataLoader(Subset(dataset, test_idx), batch_size=batch_size),
    }
    return loaders, {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    total = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)
    return 1 - dice.mean()


def class_from_masks(pred_probs: torch.Tensor, inputs: torch.Tensor) -> np.ndarray:
    occupied = inputs[:, :1]
    occupied_count = occupied.sum(dim=(1, 2, 3)).clamp_min(1)
    predicted_fraction = (pred_probs * occupied).sum(dim=(1, 2, 3)) / occupied_count
    return label_from_fraction(predicted_fraction.detach().cpu().numpy())


def run_epoch(model, loader, device, loss_fn, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "pixel_correct": 0,
        "pixel_count": 0,
        "class_correct": 0,
        "count": 0,
    }

    for inputs, masks, labels in loader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(inputs)
        bce = loss_fn(logits, masks)
        dice = dice_loss(logits, masks)
        loss = bce + dice

        if is_train:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        pred_mask = probs >= 0.5
        batch_size = labels.size(0)
        class_pred = torch.tensor(class_from_masks(probs, inputs), device=device)

        totals["loss"] += loss.item() * batch_size
        totals["dice"] += (1 - dice.item()) * batch_size
        totals["pixel_correct"] += (pred_mask == masks.bool()).sum().item()
        totals["pixel_count"] += masks.numel()
        totals["class_correct"] += (class_pred == labels).sum().item()
        totals["count"] += batch_size

    return {
        "loss": totals["loss"] / totals["count"],
        "dice": totals["dice"] / totals["count"],
        "pixel_acc": totals["pixel_correct"] / totals["pixel_count"],
        "class_acc": totals["class_correct"] / totals["count"],
    }


def predict_classes(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for inputs, _, labels in loader:
            inputs = inputs.to(device)
            probs = torch.sigmoid(model(inputs))
            predictions.extend(class_from_masks(probs, inputs).tolist())
            targets.extend(labels.numpy().tolist())

    return np.array(predictions), np.array(targets)


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(IDX_TO_LABEL), len(IDX_TO_LABEL)), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Mask Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, [row["train_dice"] for row in history], label="train")
    plt.plot(epochs, [row["val_dice"] for row in history], label="val")
    plt.title("Mask Dice")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, [row["train_class_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_class_acc"] for row in history], label="val")
    plt.title("Derived Class Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion(matrix: np.ndarray, output_path: Path) -> None:
    labels = [IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))]
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Burn-Mask U-Net Derived Class Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BurnMaskDataset(args.data_dir)
    loaders, split_sizes = make_loaders(dataset, args.batch_size, args.seed)
    device = get_device()
    model = BurnMaskUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Split sizes: {split_sizes}")

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "max_epochs reached"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, loss_fn, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_pixel_acc": train_metrics["pixel_acc"],
                "train_class_acc": train_metrics["class_acc"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_pixel_acc": val_metrics["pixel_acc"],
                "val_class_acc": val_metrics["class_acc"],
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.3f} "
            f"train_class_acc={train_metrics['class_acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.3f} "
            f"val_class_acc={val_metrics['class_acc']:.3f}",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience and epochs_without_improvement >= args.patience:
            stop_reason = (
                f"early stopped: val_loss did not improve for {args.patience} epochs"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(model, loaders["test"], device, loss_fn)
    predictions, targets = predict_classes(model, loaders["test"], device)
    matrix = confusion_matrix(predictions, targets)

    metrics = {
        "data_dir": str(args.data_dir),
        "dataset_size": len(dataset),
        "split_sizes": split_sizes,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stop_reason": stop_reason,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_pixel_acc": test_metrics["pixel_acc"],
        "test_class_acc": test_metrics["class_acc"],
        "confusion_matrix": matrix.tolist(),
        "history": history,
    }

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, args.output_dir / "training_curves.png")
    plot_confusion(matrix, args.output_dir / "confusion_matrix.png")

    print(f"Test mask loss: {test_metrics['loss']:.4f}")
    print(f"Test mask dice: {test_metrics['dice']:.3f}")
    print(f"Derived test class accuracy: {test_metrics['class_acc']:.3f}")
    print(f"Best epoch by val_loss: {best_epoch}")
    print(f"Stop reason: {stop_reason}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
