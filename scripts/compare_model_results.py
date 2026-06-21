"""Compare baseline and refined model results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = [
    ("baseline", Path("outputs/baseline_cnn/metrics.json")),
    ("multitask_0.1", Path("outputs/multitask_cnn_w01/metrics.json")),
    ("multitask_0.5", Path("outputs/multitask_cnn_w05/metrics.json")),
    ("multitask_1.0", Path("outputs/multitask_cnn/metrics.json")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/model_comparison.md"))
    return parser.parse_args()


def recall(matrix: list[list[int]], row_idx: int) -> float:
    return matrix[row_idx][row_idx] / sum(matrix[row_idx])


def summarize_run(name: str, path: Path) -> dict:
    metrics = json.loads(path.read_text())
    matrix = metrics["confusion_matrix"]
    return {
        "name": name,
        "test_acc": metrics["test_acc"],
        "test_loss": metrics["test_loss"],
        "burn_mae": metrics.get("test_burn_mae"),
        "subcritical_recall": recall(matrix, 0),
        "critical_recall": recall(matrix, 1),
        "supercritical_recall": recall(matrix, 2),
        "confusion_matrix": matrix,
    }


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# Model Comparison",
        "",
        "| Run | Test Acc | Test Loss | Burn MAE | Sub Recall | Critical Recall | Super Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        burn_mae = "n/a" if row["burn_mae"] is None else f"{row['burn_mae']:.4f}"
        lines.append(
            f"| {row['name']} | {row['test_acc']:.4f} | {row['test_loss']:.4f} | "
            f"{burn_mae} | {row['subcritical_recall']:.3f} | "
            f"{row['critical_recall']:.3f} | {row['supercritical_recall']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "The baseline CNN already performs well, and longer classification-only "
            "training plateaued quickly. The multi-task CNN provides the clearest "
            "project improvement because it adds burned-fraction prediction and "
            "raises critical-class recall, which is the most important transition "
            "region for this project. The best classification tradeoff is the "
            "`multitask_0.5` / `multitask_1.0` setting; both produce the same "
            "confusion matrix on this test split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = [summarize_run(name, path) for name, path in RUNS if path.exists()]
    markdown = build_markdown(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(markdown)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

