"""Summarize baseline CNN metrics for the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = ["subcritical", "critical", "supercritical"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/baseline_cnn/metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_cnn/results_summary.md"),
    )
    return parser.parse_args()


def class_metrics(matrix: np.ndarray) -> list[dict]:
    rows = []
    for idx, label in enumerate(LABELS):
        true_positive = matrix[idx, idx]
        false_positive = matrix[:, idx].sum() - true_positive
        false_negative = matrix[idx, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(matrix[idx, :].sum()),
            }
        )
    return rows


def build_summary(metrics: dict, rows: list[dict]) -> str:
    lines = [
        "# Baseline CNN Results Summary",
        "",
        f"Dataset size: {metrics['dataset_size']}",
        f"Training epochs: {metrics['epochs']}",
        f"Batch size: {metrics['batch_size']}",
        f"Learning rate: {metrics['learning_rate']}",
        f"Best validation accuracy: {metrics['best_val_acc']:.4f}",
        f"Test accuracy: {metrics['test_acc']:.4f}",
        f"Test loss: {metrics['test_loss']:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['label']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The classifier separates the two extreme classes well. Most mistakes involve "
            "the critical class, which is expected because it represents the transition "
            "region between fires that die out and fires that spread across most of the map.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text())
    matrix = np.array(metrics["confusion_matrix"])
    summary = build_summary(metrics, class_metrics(matrix))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary)
    print(summary)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

