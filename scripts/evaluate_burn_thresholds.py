"""Evaluate burned-fraction predictions as criticality classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parent))

from train_multitask_cnn import (  # noqa: E402
    ForestFireMultiTaskDataset,
    IDX_TO_LABEL,
    MultiTaskCriticalityCNN,
    confusion_matrix,
    get_device,
    label_from_burned_fraction,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/burn_threshold_eval")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4905)
    return parser.parse_args()


def ordered_threshold_predict(
    values: np.ndarray, low: float, high: float
) -> np.ndarray:
    labels = np.zeros(len(values), dtype=np.int64)
    labels[(values >= low) & (values < high)] = 1
    labels[values >= high] = 2
    return labels


def fit_ordered_thresholds(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    candidates = np.linspace(values.min(), values.max(), 251)
    best_low = float(candidates[0])
    best_high = float(candidates[-1])
    best_acc = -1.0

    for low_idx, low in enumerate(candidates):
        for high in candidates[low_idx:]:
            predictions = ordered_threshold_predict(values, low, high)
            acc = float(np.mean(predictions == labels))
            if acc > best_acc:
                best_low = float(low)
                best_high = float(high)
                best_acc = acc

    return best_low, best_high, best_acc


def predict_all(
    model: MultiTaskCriticalityCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    class_predictions = []
    labels = []
    burn_predictions = []
    burn_targets = []

    with torch.no_grad():
        for grids, batch_labels, burned in loader:
            logits, burned_pred = model(grids.to(device))
            class_predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch_labels.numpy().tolist())
            burn_predictions.extend(burned_pred.cpu().numpy().tolist())
            burn_targets.extend(burned.numpy().tolist())

    return (
        np.array(class_predictions),
        np.array(labels),
        np.array(burn_predictions),
        np.array(burn_targets),
        label_from_burned_fraction(np.array(burn_targets)),
    )


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


def build_markdown(metrics: dict) -> str:
    lines = [
        "# Burned-Fraction Threshold Evaluation",
        "",
        "This checks whether the regression head can recover criticality labels better "
        "than the classifier head.",
        "",
        "| Method | Accuracy | Sub Recall | Critical Recall | Super Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, row in metrics["methods"].items():
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | "
            f"{row['subcritical_recall']:.3f} | {row['critical_recall']:.3f} | "
            f"{row['supercritical_recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"Validation-tuned thresholds: low={metrics['tuned_thresholds']['low']:.4f}, "
            f"high={metrics['tuned_thresholds']['high']:.4f}.",
            "",
            f"Burned-fraction test MAE: {metrics['burn_mae']:.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestFireMultiTaskDataset(args.data_dir)
    _, val_idx, test_idx = stratified_split(dataset.labels, args.seed)
    device = get_device()

    model = MultiTaskCriticalityCNN()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size)

    _, val_labels, val_burn_pred, _, _ = predict_all(model, val_loader, device)
    class_pred, test_labels, test_burn_pred, test_burn_target, _ = predict_all(
        model, test_loader, device
    )

    low, high, val_acc = fit_ordered_thresholds(val_burn_pred, val_labels)
    physical_threshold_pred = label_from_burned_fraction(test_burn_pred)
    tuned_threshold_pred = ordered_threshold_predict(test_burn_pred, low, high)

    metrics = {
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path),
        "burn_mae": float(np.mean(np.abs(test_burn_pred - test_burn_target))),
        "burn_rmse": float(np.sqrt(np.mean((test_burn_pred - test_burn_target) ** 2))),
        "tuned_thresholds": {
            "low": low,
            "high": high,
            "validation_accuracy": val_acc,
        },
        "methods": {
            "classifier head": evaluate_predictions(class_pred, test_labels),
            "burn prediction, physical thresholds": evaluate_predictions(
                physical_threshold_pred, test_labels
            ),
            "burn prediction, tuned thresholds": evaluate_predictions(
                tuned_threshold_pred, test_labels
            ),
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    markdown_path = args.output_dir / "burn_threshold_eval.md"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    markdown = build_markdown(metrics)
    markdown_path.write_text(markdown)

    print(markdown)
    print(f"Saved {metrics_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
