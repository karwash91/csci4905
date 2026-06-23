"""Train a conditional VAE that generates forest layouts by requested class."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forest_fire_sim import (  # noqa: E402
    BURNING,
    EMPTY,
    TREE,
    label_from_burned_fraction,
    step_fire,
)

LABEL_TO_IDX = {"subcritical": 0, "critical": 1, "supercritical": 2}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

COLORS = {
    0: (0.92, 0.86, 0.70),
    1: (0.10, 0.45, 0.16),
    2: (0.95, 0.32, 0.12),
    3: (0.08, 0.08, 0.08),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/conditional_vae")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=4905)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--generated-per-class", type=int, default=40)
    parser.add_argument("--ignition-count", type=int, default=3)
    parser.add_argument(
        "--sample-mode",
        choices=("density-sample", "density-match", "threshold", "bernoulli"),
        default="density-sample",
    )
    parser.add_argument("--sample-threshold", type=float, default=0.5)
    return parser.parse_args()


class ForestLayoutDataset(Dataset):
    """Use occupied cells as the VAE image and class label as the condition."""

    def __init__(self, data_dir: Path):
        self.grids = np.load(data_dir / "grids.npy")
        self.rows = self._load_metadata(data_dir / "metadata.csv")
        self.labels = np.array([LABEL_TO_IDX[row["label"]] for row in self.rows])
        self.densities = np.array(
            [float(row["density"]) for row in self.rows], dtype=np.float32
        )

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        occupied = (self.grids[index] != EMPTY).astype(np.float32)
        image = torch.tensor(occupied, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def mean_density_by_label(self) -> dict[str, float]:
        densities = {}
        for label, label_idx in LABEL_TO_IDX.items():
            densities[label] = float(np.mean(self.densities[self.labels == label_idx]))
        return densities


class ConditionalVAE(nn.Module):
    """Small convolutional VAE conditioned on the requested criticality class."""

    def __init__(self, latent_dim: int, num_classes: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1 + num_classes, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_out = 128 * 4 * 4
        self.mu = nn.Linear(self.encoder_out, latent_dim)
        self.logvar = nn.Linear(self.encoder_out, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + num_classes, self.encoder_out)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def condition_map(self, labels: torch.Tensor, size: int) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return one_hot[:, :, None, None].expand(-1, -1, size, size)

    def encode(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = torch.cat([x, self.condition_map(labels, x.shape[-1])], dim=1)
        hidden = self.encoder(conditioned)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        hidden = self.decoder_input(torch.cat([z, one_hot], dim=1))
        return self.decoder(hidden)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


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


def make_loaders(dataset: ForestLayoutDataset, batch_size: int, seed: int):
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


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    kl_loss = kl_loss / (target.shape[-1] * target.shape[-2])
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def run_epoch(model, loader, device, beta: float, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "count": 0}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits, mu, logvar = model(images, labels)
        loss, recon_loss, kl_loss = vae_loss(logits, images, mu, logvar, beta)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["recon_loss"] += recon_loss.item() * batch_size
        totals["kl_loss"] += kl_loss.item() * batch_size
        totals["count"] += batch_size

    return {
        "loss": totals["loss"] / totals["count"],
        "recon_loss": totals["recon_loss"] / totals["count"],
        "kl_loss": totals["kl_loss"] / totals["count"],
    }


def ignite_generated_grid(
    rng: np.random.Generator,
    occupied_grid: np.ndarray,
    ignition_count: int,
) -> np.ndarray:
    grid = np.where(occupied_grid, TREE, EMPTY).astype(np.uint8)
    tree_locations = np.argwhere(grid == TREE)
    if len(tree_locations) == 0:
        return grid

    selected_count = min(ignition_count, len(tree_locations))
    selected = rng.choice(len(tree_locations), size=selected_count, replace=False)
    for row, col in tree_locations[selected]:
        grid[row, col] = BURNING
    return grid


def run_from_initial_grid(
    rng: np.random.Generator,
    initial_grid: np.ndarray,
    max_steps: int,
) -> tuple[float, str, int]:
    grid = initial_grid.copy()
    steps = 0

    while np.any(grid == BURNING) and steps < max_steps:
        grid = step_fire(grid, spread_probability=1.0, rng=rng)
        steps += 1

    tree_count = np.count_nonzero(initial_grid != EMPTY)
    burned_count = np.count_nonzero(grid == 3)
    burned_fraction = burned_count / tree_count if tree_count else 0.0
    return burned_fraction, label_from_burned_fraction(burned_fraction), steps


def generate_samples(
    model: ConditionalVAE,
    device: torch.device,
    args: argparse.Namespace,
    target_densities: dict[str, float],
) -> tuple[np.ndarray, list[dict]]:
    model.eval()
    rng = np.random.default_rng(args.seed + 99)
    grids, rows = [], []

    with torch.no_grad():
        for requested_idx, requested_label in IDX_TO_LABEL.items():
            labels = torch.full(
                (args.generated_per_class,),
                requested_idx,
                dtype=torch.long,
                device=device,
            )
            z = torch.randn(args.generated_per_class, args.latent_dim, device=device)
            probs = torch.sigmoid(model.decode(z, labels)).cpu().numpy()[:, 0]

            for sample_idx, prob_grid in enumerate(probs):
                if args.sample_mode in {"density-sample", "density-match"}:
                    target_density = target_densities[requested_label]
                    tree_count = max(1, int(round(target_density * prob_grid.size)))
                    flat_probs = prob_grid.reshape(-1).astype(float)

                    if args.sample_mode == "density-sample":
                        weights = flat_probs / flat_probs.sum()
                        selected = rng.choice(
                            prob_grid.size,
                            size=tree_count,
                            replace=False,
                            p=weights,
                        )
                    else:
                        selected = np.argpartition(flat_probs, -tree_count)[
                            -tree_count:
                        ]

                    occupied = np.zeros(prob_grid.size, dtype=bool)
                    occupied[selected] = True
                    occupied = occupied.reshape(prob_grid.shape)
                elif args.sample_mode == "bernoulli":
                    occupied = rng.random(prob_grid.shape) < prob_grid
                else:
                    occupied = prob_grid >= args.sample_threshold

                initial_grid = ignite_generated_grid(rng, occupied, args.ignition_count)
                burned_fraction, simulated_label, steps = run_from_initial_grid(
                    rng,
                    initial_grid,
                    max_steps=initial_grid.shape[0] * 8,
                )
                generated_density = (
                    np.count_nonzero(initial_grid != EMPTY) / initial_grid.size
                )

                grids.append(initial_grid)
                rows.append(
                    {
                        "sample_id": len(rows),
                        "requested_label": requested_label,
                        "requested_label_idx": requested_idx,
                        "sample_idx": sample_idx,
                        "density": generated_density,
                        "burned_fraction": burned_fraction,
                        "simulated_label": simulated_label,
                        "match": requested_label == simulated_label,
                        "steps": steps,
                    }
                )

    return np.stack(grids).astype(np.uint8), rows


def colorize(grid: np.ndarray) -> np.ndarray:
    image = np.zeros((*grid.shape, 3), dtype=float)
    for value, color in COLORS.items():
        image[grid == value] = color
    return image


def plot_generated_examples(
    grids: np.ndarray, rows: list[dict], output_path: Path
) -> None:
    labels = ["subcritical", "critical", "supercritical"]
    fig, axes = plt.subplots(len(labels), 5, figsize=(11, 7))

    for row_idx, requested_label in enumerate(labels):
        examples = [row for row in rows if row["requested_label"] == requested_label][
            :5
        ]
        for col_idx, row in enumerate(examples):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            grid = grids[int(row["sample_id"])]
            ax.imshow(colorize(grid))
            ax.set_title(
                f"ask: {requested_label}\n"
                f"sim: {row['simulated_label']}\n"
                f"burn={float(row['burned_fraction']):.2f}",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_history(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("VAE Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_recon_loss"] for row in history], label="train recon")
    plt.plot(epochs, [row["val_recon_loss"] for row in history], label="val recon")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_match_matrix(rows: list[dict], output_path: Path) -> np.ndarray:
    matrix = np.zeros((len(LABEL_TO_IDX), len(LABEL_TO_IDX)), dtype=int)
    for row in rows:
        requested_idx = LABEL_TO_IDX[row["requested_label"]]
        simulated_idx = LABEL_TO_IDX[row["simulated_label"]]
        matrix[requested_idx, simulated_idx] += 1

    labels = [IDX_TO_LABEL[idx] for idx in range(len(IDX_TO_LABEL))]
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Greens")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Simulator Result")
    plt.ylabel("Requested Class")
    plt.title("Generated Sample Check")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(matrix[row_idx, col_idx]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return matrix


def write_generated_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_generated(rows: list[dict]) -> dict:
    summary = {}
    for requested_label in LABEL_TO_IDX:
        label_rows = [row for row in rows if row["requested_label"] == requested_label]
        matches = sum(bool(row["match"]) for row in label_rows)
        summary[requested_label] = {
            "count": len(label_rows),
            "match_rate": matches / len(label_rows) if label_rows else 0.0,
            "mean_density": float(
                np.mean([float(row["density"]) for row in label_rows])
            ),
            "mean_burned_fraction": float(
                np.mean([float(row["burned_fraction"]) for row in label_rows])
            ),
        }
    summary["overall_match_rate"] = sum(bool(row["match"]) for row in rows) / len(rows)
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestLayoutDataset(args.data_dir)
    loaders, split_sizes = make_loaders(dataset, args.batch_size, args.seed)
    device = get_device()

    model = ConditionalVAE(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Split sizes: {split_sizes}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, args.beta, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device, args.beta)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_recon_loss": train_metrics["recon_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "val_loss": val_metrics["loss"],
                "val_recon_loss": val_metrics["recon_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_recon={val_metrics['recon_loss']:.4f} "
            f"val_kl={val_metrics['kl_loss']:.4f}",
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
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(model, loaders["test"], device, args.beta)
    target_densities = dataset.mean_density_by_label()
    generated_grids, generated_rows = generate_samples(
        model, device, args, target_densities
    )
    generated_summary = summarize_generated(generated_rows)
    match_matrix = plot_match_matrix(
        generated_rows, args.output_dir / "generated_match_matrix.png"
    )

    np.save(args.output_dir / "generated_grids.npy", generated_grids)
    write_generated_metadata(args.output_dir / "generated_metadata.csv", generated_rows)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    plot_history(history, args.output_dir / "training_curves.png")
    plot_generated_examples(
        generated_grids, generated_rows, args.output_dir / "generated_examples.png"
    )

    metrics = {
        "dataset_size": len(dataset),
        "split_sizes": split_sizes,
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_recon_loss": test_metrics["recon_loss"],
        "test_kl_loss": test_metrics["kl_loss"],
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "sample_threshold": args.sample_threshold,
        "sample_mode": args.sample_mode,
        "target_densities": target_densities,
        "generated_per_class": args.generated_per_class,
        "generated_summary": generated_summary,
        "generated_match_matrix": match_matrix.tolist(),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Generated overall match rate: {generated_summary['overall_match_rate']:.3f}")
    print(f"Generated summary: {generated_summary}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
