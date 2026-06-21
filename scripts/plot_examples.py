"""Plot example generated grids by criticality label."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

MPL_CACHE_DIR = Path("outputs/.matplotlib").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["subcritical", "critical", "supercritical"]
COLORS = {
    0: (0.92, 0.86, 0.70),
    1: (0.10, 0.45, 0.16),
    2: (0.95, 0.32, 0.12),
    3: (0.08, 0.08, 0.08),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output", type=Path, default=Path("outputs/example_grids.png"))
    return parser.parse_args()


def colorize(grid: np.ndarray) -> np.ndarray:
    image = np.zeros((*grid.shape, 3), dtype=float)
    for value, color in COLORS.items():
        image[grid == value] = color
    return image


def load_metadata(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    args = parse_args()
    grids = np.load(args.data_dir / "grids.npy")
    metadata = load_metadata(args.data_dir / "metadata.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(LABELS), 4, figsize=(10, 7))
    for row_idx, label in enumerate(LABELS):
        examples = [row for row in metadata if row["label"] == label][:4]
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(examples):
                continue

            sample = examples[col_idx]
            sample_id = int(sample["sample_id"])
            ax.imshow(colorize(grids[sample_id]))
            ax.set_title(
                f"{label}\n"
                f"d={float(sample['density']):.2f}, "
                f"burn={float(sample['burned_fraction']):.2f}",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
