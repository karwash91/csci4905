"""Simple NumPy forest-fire simulator for synthetic criticality data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3


@dataclass(frozen=True)
class SimulationResult:
    """Container for one completed forest-fire simulation."""

    initial_grid: np.ndarray
    final_grid: np.ndarray
    density: float
    burned_fraction: float
    label: str
    steps: int


def label_from_burned_fraction(burned_fraction: float) -> str:
    """Map final burned fraction to a coarse criticality label."""
    if burned_fraction < 0.10:
        return "subcritical"
    if burned_fraction < 0.90:
        return "critical"
    return "supercritical"


def make_initial_grid(
    rng: np.random.Generator,
    size: int = 128,
    density: float = 0.60,
    ignition_count: int = 3,
) -> np.ndarray:
    """Create a random tree grid and ignite a few occupied cells."""
    grid = np.where(rng.random((size, size)) < density, TREE, EMPTY).astype(np.uint8)
    tree_locations = np.argwhere(grid == TREE)

    if len(tree_locations) == 0:
        return grid

    ignition_count = min(ignition_count, len(tree_locations))
    selected = rng.choice(len(tree_locations), size=ignition_count, replace=False)
    for row, col in tree_locations[selected]:
        grid[row, col] = BURNING

    return grid


def burning_neighbor_mask(grid: np.ndarray) -> np.ndarray:
    """Return cells adjacent to a burning cell using 4-neighborhood spread."""
    burning = grid == BURNING
    mask = np.zeros_like(burning, dtype=bool)
    mask[1:, :] |= burning[:-1, :]
    mask[:-1, :] |= burning[1:, :]
    mask[:, 1:] |= burning[:, :-1]
    mask[:, :-1] |= burning[:, 1:]
    return mask


def step_fire(grid: np.ndarray, spread_probability: float, rng: np.random.Generator) -> np.ndarray:
    """Advance the simulation by one step."""
    next_grid = grid.copy()
    next_grid[grid == BURNING] = BURNED

    candidates = (grid == TREE) & burning_neighbor_mask(grid)
    ignites = candidates & (rng.random(grid.shape) < spread_probability)
    next_grid[ignites] = BURNING
    return next_grid


def run_simulation(
    rng: np.random.Generator,
    size: int = 128,
    density: float = 0.60,
    ignition_count: int = 3,
    spread_probability: float = 1.0,
    max_steps: int = 512,
) -> SimulationResult:
    """Run one forest-fire simulation until no cells are burning."""
    initial_grid = make_initial_grid(rng, size, density, ignition_count)
    grid = initial_grid.copy()
    steps = 0

    while np.any(grid == BURNING) and steps < max_steps:
        grid = step_fire(grid, spread_probability, rng)
        steps += 1

    initial_tree_count = np.count_nonzero(initial_grid != EMPTY)
    burned_count = np.count_nonzero(grid == BURNED)
    burned_fraction = burned_count / initial_tree_count if initial_tree_count else 0.0

    return SimulationResult(
        initial_grid=initial_grid,
        final_grid=grid,
        density=density,
        burned_fraction=burned_fraction,
        label=label_from_burned_fraction(burned_fraction),
        steps=steps,
    )
