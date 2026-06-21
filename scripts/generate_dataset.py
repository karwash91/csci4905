"""Generate a synthetic forest-fire dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forest_fire_sim import run_simulation


DEFAULT_DENSITIES = [0.30, 0.40, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4905)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grids = []
    rows = []

    for idx in range(args.samples):
        base_density = DEFAULT_DENSITIES[idx % len(DEFAULT_DENSITIES)]
        density = float(np.clip(base_density + rng.normal(0, 0.02), 0.05, 0.95))
        result = run_simulation(rng=rng, size=args.size, density=density)

        grids.append(result.initial_grid)
        rows.append(
            {
                "sample_id": idx,
                "density": result.density,
                "burned_fraction": result.burned_fraction,
                "label": result.label,
                "steps": result.steps,
            }
        )

    grids_array = np.stack(grids).astype(np.uint8)
    np.save(args.output_dir / "grids.npy", grids_array)

    with (args.output_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    label_counts = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"Saved {len(rows)} samples to {args.output_dir}")
    print(f"Grid shape: {grids_array.shape}")
    print(f"Label counts: {label_counts}")


if __name__ == "__main__":
    main()
