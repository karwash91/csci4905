"""Train a baseline CNN to classify forest-fire criticality."""

from __future__ import annotations

import argparse
import csv
import json
import os
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


class ForestFireDataset(Dataset):
    """Load generated grids and metadata from disk."""

    def __init__(self, data_dir: Path):
        self.grids = np.load(data_dir / "grids.npy")
        self.rows = self._load_metadata(data_dir / "metadata.csv")
        self.labels = np.array([LABEL_TO_IDX[row["label"]] for row in self.rows])

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.tensor(self.grids[index], dtype=torch.float32).unsqueeze(0) / 3.0
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return grid, label


class CriticalityCNN(nn.Module):
    """Small CNN baseline for 2D cellular automaton grids."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline_cnn"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=4905)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split(labels: np.ndarray, seed: int) -> tuple[list[int], list[int], list[int]]:
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


def make_loaders(dataset: ForestFireDataset, batch_size: int, seed: int):
    train_idx, val_idx, test_idx = stratified_split(dataset.labels, seed)
    loaders = {
        "train": DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True),
        "val": DataLoader(Subset(dataset, val_idx), batch_size=batch_size),
        "test": DataLoader(Subset(dataset, test_idx), batch_size=batch_size),
    }
    split_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    return loaders, split_sizes


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for grids, labels in loader:
        grids = grids.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(grids)
        loss = loss_fn(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for grids, labels in loader:
            logits = model(grids.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())

    return np.array(predictions), np.array(targets)


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABEL_TO_IDX), len(LABEL_TO_IDX)), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.title("Accuracy")
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
    plt.title("Test Confusion Matrix")

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

    dataset = ForestFireDataset(args.data_dir)
    loaders, split_sizes = make_loaders(dataset, args.batch_size, args.seed)
    device = get_device()

    model = CriticalityCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Split sizes: {split_sizes}")

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], loss_fn, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, loaders["test"], loss_fn, device)
    predictions, targets = predict_all(model, loaders["test"], device)
    matrix = confusion_matrix(predictions, targets)

    metrics = {
        "dataset_size": len(dataset),
        "split_sizes": split_sizes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": matrix.tolist(),
    }

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, args.output_dir / "training_curves.png")
    plot_confusion(matrix, args.output_dir / "confusion_matrix.png")

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
