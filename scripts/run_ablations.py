"""Run simple ablations for the forest-fire criticality model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))

from train_cnn import (  # noqa: E402
    CriticalityCNN,
    ForestFireDataset,
    IDX_TO_LABEL,
    confusion_matrix,
    get_device,
    stratified_split,
)


class ShuffledGridDataset(Dataset):
    """Preserve each grid's cell counts, but destroy spatial arrangement."""

    def __init__(self, dataset: ForestFireDataset, indices: list[int], seed: int):
        self.dataset = dataset
        self.indices = indices
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = self.indices[index]
        grid = self.dataset.grids[source_index].reshape(-1)
        rng = np.random.default_rng(self.seed + int(source_index))
        shuffled = grid[rng.permutation(grid.size)].reshape(
            self.dataset.grids.shape[1:]
        )

        tensor = torch.tensor(shuffled, dtype=torch.float32).unsqueeze(0) / 3.0
        label = torch.tensor(self.dataset.labels[source_index], dtype=torch.long)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("outputs/baseline_cnn/model.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4905)
    return parser.parse_args()


def grid_density(grids: np.ndarray) -> np.ndarray:
    """Density visible from the model input: non-empty cells divided by grid size."""
    return np.count_nonzero(grids != 0, axis=(1, 2)) / np.prod(grids.shape[1:])


def ordered_threshold_predict(
    values: np.ndarray, low: float, high: float
) -> np.ndarray:
    predictions = np.zeros(len(values), dtype=np.int64)
    predictions[(values >= low) & (values < high)] = 1
    predictions[values >= high] = 2
    return predictions


def fit_density_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: list[int],
) -> tuple[float, float, float]:
    """Fit two thresholds that map density to three ordered classes."""
    train_values = values[train_indices]
    train_labels = labels[train_indices]
    candidates = np.linspace(train_values.min(), train_values.max(), 251)

    best_low = float(candidates[0])
    best_high = float(candidates[-1])
    best_acc = -1.0

    for low_idx, low in enumerate(candidates):
        for high in candidates[low_idx:]:
            predictions = ordered_threshold_predict(train_values, low, high)
            acc = float(np.mean(predictions == train_labels))
            if acc > best_acc:
                best_low = float(low)
                best_high = float(high)
                best_acc = acc

    return best_low, best_high, best_acc


def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray) -> dict:
    matrix = confusion_matrix(predictions, targets)
    recalls = {}
    for idx, label in IDX_TO_LABEL.items():
        row_total = matrix[idx].sum()
        recalls[f"{label}_recall"] = float(matrix[idx, idx] / row_total)

    return {
        "accuracy": float(np.mean(predictions == targets)),
        "confusion_matrix": matrix.tolist(),
        **recalls,
    }


def predict_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    predictions = []

    with torch.no_grad():
        for grids, _ in loader:
            logits = model(grids.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return np.array(predictions)


def build_markdown(metrics: dict) -> str:
    density = metrics["density_only"]
    normal = metrics.get("cnn_normal_test")
    shuffled = metrics.get("cnn_shuffled_test")

    lines = [
        "# Ablation Results",
        "",
        "These checks test whether the CNN is doing more than reading tree density.",
        "",
        "| Check | Accuracy | Sub Recall | Critical Recall | Super Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Density-only thresholds | {density['accuracy']:.4f} | "
            f"{density['subcritical_recall']:.3f} | {density['critical_recall']:.3f} | "
            f"{density['supercritical_recall']:.3f} |"
        ),
    ]

    if normal is not None:
        lines.append(
            f"| CNN on normal test grids | {normal['accuracy']:.4f} | "
            f"{normal['subcritical_recall']:.3f} | {normal['critical_recall']:.3f} | "
            f"{normal['supercritical_recall']:.3f} |"
        )

    if shuffled is not None:
        lines.append(
            f"| CNN on shuffled test grids | {shuffled['accuracy']:.4f} | "
            f"{shuffled['subcritical_recall']:.3f} | {shuffled['critical_recall']:.3f} | "
            f"{shuffled['supercritical_recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Density-only thresholds use one scalar feature: the fraction of non-empty cells "
            "in the initial grid. Shuffled grids keep the same density and cell values as each "
            "test sample, but randomize their positions.",
            "",
            "If shuffled-grid performance is close to normal CNN performance, density explains "
            "most of the model result. If shuffled-grid performance drops, the CNN is using "
            "some spatial information from the grid layout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestFireDataset(args.data_dir)
    train_idx, _, test_idx = stratified_split(dataset.labels, args.seed)
    test_targets = dataset.labels[test_idx]

    densities = grid_density(dataset.grids)
    low, high, train_acc = fit_density_thresholds(densities, dataset.labels, train_idx)
    density_predictions = ordered_threshold_predict(densities[test_idx], low, high)

    metrics = {
        "density_only": {
            "low_threshold": low,
            "high_threshold": high,
            "train_accuracy": train_acc,
            **evaluate_predictions(density_predictions, test_targets),
        }
    }

    if args.model_path.exists():
        device = get_device()
        model = CriticalityCNN()
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)

        normal_loader = DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch_size,
        )
        shuffled_loader = DataLoader(
            ShuffledGridDataset(dataset, test_idx, args.seed),
            batch_size=args.batch_size,
        )

        normal_predictions = predict_model(model, normal_loader, device)
        shuffled_predictions = predict_model(model, shuffled_loader, device)

        metrics["cnn_normal_test"] = evaluate_predictions(
            normal_predictions, test_targets
        )
        metrics["cnn_shuffled_test"] = evaluate_predictions(
            shuffled_predictions, test_targets
        )

    metrics_path = args.output_dir / "metrics.json"
    markdown_path = args.output_dir / "ablation_results.md"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    markdown = build_markdown(metrics)
    markdown_path.write_text(markdown)

    print(markdown)
    print(f"Saved {metrics_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
