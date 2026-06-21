"""Train a CNN with criticality classification and burned-fraction regression."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
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


class ForestFireMultiTaskDataset(Dataset):
    """Load grids, class labels, and continuous burned fraction."""

    def __init__(self, data_dir: Path):
        self.grids = np.load(data_dir / "grids.npy")
        self.rows = self._load_metadata(data_dir / "metadata.csv")
        self.labels = np.array([LABEL_TO_IDX[row["label"]] for row in self.rows])
        self.burned_fractions = np.array(
            [float(row["burned_fraction"]) for row in self.rows], dtype=np.float32
        )

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = torch.tensor(self.grids[index], dtype=torch.float32).unsqueeze(0) / 3.0
        label = torch.tensor(self.labels[index], dtype=torch.long)
        burned_fraction = torch.tensor(self.burned_fractions[index], dtype=torch.float32)
        return grid, label, burned_fraction


class MultiTaskCriticalityCNN(nn.Module):
    """CNN with one shared encoder and two prediction heads."""

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
            nn.Flatten(),
        )
        self.shared = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(32, num_classes)
        self.burn_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(self.features(x))
        class_logits = self.class_head(hidden)
        burned_fraction = self.burn_head(hidden).squeeze(1)
        return class_logits, burned_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multitask_cnn"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--burn-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4905)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
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


def make_loaders(dataset: ForestFireMultiTaskDataset, batch_size: int, seed: int):
    train_idx, val_idx, test_idx = stratified_split(dataset.labels, seed)
    loaders = {
        "train": DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True),
        "val": DataLoader(Subset(dataset, val_idx), batch_size=batch_size),
        "test": DataLoader(Subset(dataset, test_idx), batch_size=batch_size),
    }
    split_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    return loaders, split_sizes


def label_from_burned_fraction(values: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(values, dtype=np.int64)
    labels[(values >= 0.10) & (values < 0.90)] = LABEL_TO_IDX["critical"]
    labels[values >= 0.90] = LABEL_TO_IDX["supercritical"]
    return labels


def multitask_loss(
    class_logits: torch.Tensor,
    burned_pred: torch.Tensor,
    labels: torch.Tensor,
    burned_target: torch.Tensor,
    class_loss_fn: nn.Module,
    burn_loss_fn: nn.Module,
    burn_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_loss = class_loss_fn(class_logits, labels)
    burn_loss = burn_loss_fn(burned_pred, burned_target)
    total_loss = class_loss + burn_loss_weight * burn_loss
    return total_loss, class_loss, burn_loss


def run_epoch(
    model,
    loader,
    class_loss_fn,
    burn_loss_fn,
    device,
    burn_loss_weight: float,
    optimizer=None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "class_loss": 0.0,
        "burn_loss": 0.0,
        "correct": 0,
        "burn_abs_error": 0.0,
        "count": 0,
    }

    for grids, labels, burned_targets in loader:
        grids = grids.to(device)
        labels = labels.to(device)
        burned_targets = burned_targets.to(device)

        if is_train:
            optimizer.zero_grad()

        class_logits, burned_pred = model(grids)
        loss, class_loss, burn_loss = multitask_loss(
            class_logits,
            burned_pred,
            labels,
            burned_targets,
            class_loss_fn,
            burn_loss_fn,
            burn_loss_weight,
        )

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["class_loss"] += class_loss.item() * batch_size
        totals["burn_loss"] += burn_loss.item() * batch_size
        totals["correct"] += (class_logits.argmax(dim=1) == labels).sum().item()
        totals["burn_abs_error"] += torch.abs(burned_pred - burned_targets).sum().item()
        totals["count"] += batch_size

    return {
        "loss": totals["loss"] / totals["count"],
        "class_loss": totals["class_loss"] / totals["count"],
        "burn_loss": totals["burn_loss"] / totals["count"],
        "acc": totals["correct"] / totals["count"],
        "burn_mae": totals["burn_abs_error"] / totals["count"],
    }


def is_better_loss(current_loss: float, best_loss: float, min_delta: float) -> bool:
    return current_loss < best_loss - min_delta


def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets, burned_predictions, burned_targets = [], [], [], []

    with torch.no_grad():
        for grids, labels, burned in loader:
            class_logits, burned_pred = model(grids.to(device))
            predictions.extend(class_logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
            burned_predictions.extend(burned_pred.cpu().numpy().tolist())
            burned_targets.extend(burned.numpy().tolist())

    return (
        np.array(predictions),
        np.array(targets),
        np.array(burned_predictions),
        np.array(burned_targets),
    )


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABEL_TO_IDX), len(LABEL_TO_IDX)), dtype=int)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val")
    plt.title("Class Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, [row["train_burn_mae"] for row in history], label="train")
    plt.plot(epochs, [row["val_burn_mae"] for row in history], label="val")
    plt.title("Burned Fraction MAE")
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
    plt.title("Multi-Task Test Confusion Matrix")

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

    dataset = ForestFireMultiTaskDataset(args.data_dir)
    loaders, split_sizes = make_loaders(dataset, args.batch_size, args.seed)
    device = get_device()

    model = MultiTaskCriticalityCNN().to(device)
    class_loss_fn = nn.CrossEntropyLoss()
    burn_loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Split sizes: {split_sizes}")
    print(f"Burn loss weight: {args.burn_loss_weight}")

    start_time = time.perf_counter()
    best_epoch = 0
    best_val_loss = float("inf")
    best_state = None
    epochs_without_loss_improvement = 0
    stop_reason = "max_epochs reached"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            class_loss_fn,
            burn_loss_fn,
            device,
            args.burn_loss_weight,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            class_loss_fn,
            burn_loss_fn,
            device,
            args.burn_loss_weight,
        )
        scheduler.step(val_metrics["loss"])
        elapsed_seconds = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_class_loss": train_metrics["class_loss"],
                "train_burn_loss": train_metrics["burn_loss"],
                "train_burn_mae": train_metrics["burn_mae"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_class_loss": val_metrics["class_loss"],
                "val_burn_loss": val_metrics["burn_loss"],
                "val_burn_mae": val_metrics["burn_mae"],
                "lr": current_lr,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"train_burn_mae={train_metrics['burn_mae']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"val_burn_mae={val_metrics['burn_mae']:.3f} "
            f"lr={current_lr:.2e} elapsed={elapsed_seconds / 60:.1f}m",
            flush=True,
        )

        if is_better_loss(val_metrics["loss"], best_val_loss, args.min_delta):
            best_epoch = epoch
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        if args.patience and epochs_without_loss_improvement >= args.patience:
            stop_reason = f"early stopped: val_loss did not improve for {args.patience} epochs"
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_seconds = time.perf_counter() - start_time
    test_metrics = run_epoch(
        model,
        loaders["test"],
        class_loss_fn,
        burn_loss_fn,
        device,
        args.burn_loss_weight,
    )
    predictions, targets, burned_predictions, burned_targets = predict_all(
        model, loaders["test"], device
    )
    matrix = confusion_matrix(predictions, targets)

    derived_predictions = label_from_burned_fraction(burned_predictions)
    derived_matrix = confusion_matrix(derived_predictions, targets)
    burned_rmse = float(np.sqrt(np.mean((burned_predictions - burned_targets) ** 2)))

    metrics = {
        "dataset_size": len(dataset),
        "split_sizes": split_sizes,
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "burn_loss_weight": args.burn_loss_weight,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_seconds,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_class_loss": test_metrics["class_loss"],
        "test_burn_loss": test_metrics["burn_loss"],
        "test_acc": test_metrics["acc"],
        "test_burn_mae": test_metrics["burn_mae"],
        "test_burn_rmse": burned_rmse,
        "confusion_matrix": matrix.tolist(),
        "burn_fraction_threshold_confusion_matrix": derived_matrix.tolist(),
        "history": history,
    }

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, args.output_dir / "training_curves.png")
    plot_confusion(matrix, args.output_dir / "confusion_matrix.png")

    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['acc']:.3f}")
    print(f"Test burned-fraction MAE: {test_metrics['burn_mae']:.4f}")
    print(f"Test burned-fraction RMSE: {burned_rmse:.4f}")
    print(f"Best epoch by val_loss: {best_epoch}")
    print(f"Stop reason: {stop_reason}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

