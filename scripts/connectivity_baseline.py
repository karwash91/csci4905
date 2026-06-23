"""Evaluate interpretable connectivity baselines for forest-fire criticality."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from train_cnn import (  # noqa: E402
    ForestFireDataset,
    IDX_TO_LABEL,
    confusion_matrix,
    stratified_split,
)

SUBCRITICAL = 0
CRITICAL = 1
SUPERCRITICAL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/connectivity_baseline")
    )
    parser.add_argument("--seed", type=int, default=4905)
    return parser.parse_args()


def ordered_threshold_predict(
    values: np.ndarray, low: float, high: float
) -> np.ndarray:
    predictions = np.zeros(len(values), dtype=np.int64)
    predictions[(values >= low) & (values < high)] = CRITICAL
    predictions[values >= high] = SUPERCRITICAL
    return predictions


def fit_ordered_thresholds(
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: list[int],
) -> tuple[float, float, float]:
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


def label_from_burn_fraction(values: np.ndarray) -> np.ndarray:
    return ordered_threshold_predict(values, low=0.10, high=0.90)


def component_features(grid: np.ndarray) -> tuple[float, float, float]:
    """Return density, largest component fraction, and ignition-reachable fraction."""
    occupied = grid != 0
    burning = grid == 2
    occupied_count = int(occupied.sum())

    if occupied_count == 0:
        return 0.0, 0.0, 0.0

    visited = np.zeros_like(occupied, dtype=bool)
    largest_component = 0
    reachable_from_ignition = 0
    rows, cols = occupied.shape

    for start_row, start_col in np.argwhere(occupied):
        if visited[start_row, start_col]:
            continue

        queue = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        component_size = 0
        touches_ignition = False

        while queue:
            row, col = queue.popleft()
            component_size += 1
            touches_ignition = touches_ignition or bool(burning[row, col])

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

        largest_component = max(largest_component, component_size)
        if touches_ignition:
            reachable_from_ignition += component_size

    density = occupied_count / grid.size
    largest_fraction = largest_component / occupied_count
    reachable_fraction = reachable_from_ignition / occupied_count
    return density, largest_fraction, reachable_fraction


def compute_features(grids: np.ndarray) -> dict[str, np.ndarray]:
    rows = [component_features(grid) for grid in grids]
    values = np.array(rows, dtype=np.float64)
    return {
        "density": values[:, 0],
        "largest_component_fraction": values[:, 1],
        "ignition_reachable_fraction": values[:, 2],
    }


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


def evaluate_threshold_feature(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: list[int],
    test_indices: list[int],
) -> dict:
    low, high, train_acc = fit_ordered_thresholds(values, labels, train_indices)
    test_predictions = ordered_threshold_predict(values[test_indices], low, high)
    return {
        "feature": name,
        "low_threshold": low,
        "high_threshold": high,
        "train_accuracy": train_acc,
        **evaluate_predictions(test_predictions, labels[test_indices]),
    }


def build_markdown(metrics: dict) -> str:
    lines = [
        "# Connectivity Baseline",
        "",
        "These baselines test whether hand-built spatial connectivity features explain the labels.",
        "",
        "| Baseline | Accuracy | Sub Recall | Critical Recall | Super Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, row in metrics["baselines"].items():
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | "
            f"{row['subcritical_recall']:.3f} | {row['critical_recall']:.3f} | "
            f"{row['supercritical_recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The largest-component baseline uses the biggest connected group of trees "
            "as a fraction of all occupied cells. The ignition-reachable baseline uses "
            "the fraction of occupied cells connected to the initially burning cells.",
            "",
            "The ignition-reachable baseline is rule-aware: with deterministic 4-neighbor "
            "spread, it closely matches what the simulator will burn.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestFireDataset(args.data_dir)
    train_idx, _, test_idx = stratified_split(dataset.labels, args.seed)
    labels = dataset.labels
    features = compute_features(dataset.grids)

    baselines = {
        "density-only thresholds": evaluate_threshold_feature(
            "density",
            features["density"],
            labels,
            train_idx,
            test_idx,
        ),
        "largest-component thresholds": evaluate_threshold_feature(
            "largest_component_fraction",
            features["largest_component_fraction"],
            labels,
            train_idx,
            test_idx,
        ),
        "ignition-reachable rule": evaluate_predictions(
            label_from_burn_fraction(features["ignition_reachable_fraction"][test_idx]),
            labels[test_idx],
        ),
    }

    metrics = {
        "data_dir": str(args.data_dir),
        "dataset_size": len(dataset),
        "baselines": baselines,
        "feature_summary": {
            name: {
                "min": float(values.min()),
                "mean": float(values.mean()),
                "max": float(values.max()),
            }
            for name, values in features.items()
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    markdown_path = args.output_dir / "connectivity_baseline.md"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    markdown = build_markdown(metrics)
    markdown_path.write_text(markdown)

    print(markdown)
    print(f"Saved {metrics_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
