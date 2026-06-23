"""Generate a fixed-density dataset where spatial layout matters."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forest_fire_sim import (  # noqa: E402
    BURNED,
    BURNING,
    EMPTY,
    TREE,
    label_from_burned_fraction,
    step_fire,
)

LAYOUT_MODES = ("random", "clustered", "fragmented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.55)
    parser.add_argument("--ignition-count", type=int, default=3)
    parser.add_argument("--spread-probability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4905)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/spatial_64"))
    return parser.parse_args()


def exact_tree_count(size: int, density: float) -> int:
    return int(round(size * size * density))


def smooth_field(field: np.ndarray, steps: int) -> np.ndarray:
    smoothed = field.astype(float)
    for _ in range(steps):
        smoothed = (
            smoothed
            + np.roll(smoothed, 1, axis=0)
            + np.roll(smoothed, -1, axis=0)
            + np.roll(smoothed, 1, axis=1)
            + np.roll(smoothed, -1, axis=1)
        ) / 5.0
    return smoothed


def grid_from_scores(scores: np.ndarray, tree_count: int) -> np.ndarray:
    selected = np.argpartition(scores.reshape(-1), -tree_count)[-tree_count:]
    grid = np.zeros(scores.size, dtype=np.uint8)
    grid[selected] = TREE
    return grid.reshape(scores.shape)


def make_spatial_grid(
    rng: np.random.Generator,
    size: int,
    density: float,
    mode: str,
) -> np.ndarray:
    tree_count = exact_tree_count(size, density)

    if mode == "random":
        scores = rng.random((size, size))
        return grid_from_scores(scores, tree_count)

    if mode == "clustered":
        scores = smooth_field(rng.random((size, size)), steps=8)
        return grid_from_scores(scores, tree_count)

    if mode == "fragmented":
        base = rng.random((size, size))
        occupied = base < min(0.95, density + 0.20)
        neighbor_count = (
            np.roll(occupied, 1, axis=0)
            + np.roll(occupied, -1, axis=0)
            + np.roll(occupied, 1, axis=1)
            + np.roll(occupied, -1, axis=1)
        )
        scores = rng.random((size, size)) - 0.25 * neighbor_count
        return grid_from_scores(scores, tree_count)

    raise ValueError(f"Unknown layout mode: {mode}")


def ignite_grid(
    rng: np.random.Generator,
    grid: np.ndarray,
    ignition_count: int,
) -> np.ndarray:
    initial_grid = grid.copy()
    tree_locations = np.argwhere(initial_grid == TREE)
    selected_count = min(ignition_count, len(tree_locations))

    if selected_count:
        selected = rng.choice(len(tree_locations), size=selected_count, replace=False)
        for row, col in tree_locations[selected]:
            initial_grid[row, col] = BURNING

    return initial_grid


def run_from_initial_grid(
    rng: np.random.Generator,
    initial_grid: np.ndarray,
    spread_probability: float,
    max_steps: int,
) -> tuple[np.ndarray, float, str, int]:
    grid = initial_grid.copy()
    steps = 0

    while np.any(grid == BURNING) and steps < max_steps:
        grid = step_fire(grid, spread_probability, rng)
        steps += 1

    initial_tree_count = np.count_nonzero(initial_grid != EMPTY)
    burned_count = np.count_nonzero(grid == BURNED)
    burned_fraction = burned_count / initial_tree_count if initial_tree_count else 0.0
    return grid, burned_fraction, label_from_burned_fraction(burned_fraction), steps


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_counts = {
        "subcritical": args.samples_per_class,
        "critical": args.samples_per_class,
        "supercritical": args.samples_per_class,
    }
    label_counts = {label: 0 for label in target_counts}
    mode_counts = {mode: 0 for mode in LAYOUT_MODES}
    grids = []
    rows = []

    attempts = 0
    while any(label_counts[label] < target_counts[label] for label in target_counts):
        attempts += 1
        if attempts > args.max_attempts:
            raise RuntimeError(
                f"Reached {args.max_attempts} attempts before filling all classes: {label_counts}"
            )

        mode = LAYOUT_MODES[attempts % len(LAYOUT_MODES)]
        tree_grid = make_spatial_grid(rng, args.size, args.density, mode)
        initial_grid = ignite_grid(rng, tree_grid, args.ignition_count)
        _, burned_fraction, label, steps = run_from_initial_grid(
            rng,
            initial_grid,
            spread_probability=args.spread_probability,
            max_steps=args.size * 8,
        )

        if label_counts[label] >= target_counts[label]:
            continue

        sample_id = len(rows)
        grids.append(initial_grid)
        rows.append(
            {
                "sample_id": sample_id,
                "density": args.density,
                "layout_mode": mode,
                "burned_fraction": burned_fraction,
                "label": label,
                "steps": steps,
            }
        )
        label_counts[label] += 1
        mode_counts[mode] += 1

    grids_array = np.stack(grids).astype(np.uint8)
    np.save(args.output_dir / "grids.npy", grids_array)

    with (args.output_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} samples to {args.output_dir}")
    print(f"Attempts: {attempts}")
    print(f"Grid shape: {grids_array.shape}")
    print(f"Density: {args.density}")
    print(f"Label counts: {label_counts}")
    print(f"Layout mode counts: {mode_counts}")


if __name__ == "__main__":
    main()
