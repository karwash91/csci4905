# Final Project Log

This document records the significant steps, decisions, and results for the final project.

## Project Goal

Build a synthetic forest-fire criticality dataset, label generated samples by fire-spread behavior, and train a neural network to identify subcritical, critical, and supercritical cases.

## Step Log

### 2026-06-21 - Project Start

Decision:
- Start with a simple local NumPy cellular automaton instead of immediately depending on `gym_forestfire`.
- Prioritize an end-to-end baseline: generate labeled grids, visualize examples, then train a small CNN.
- Treat VAE/latent-space analysis as an optional extension after the classifier works.

Reasoning:
- The main risk is label quality and data generation, not model architecture.
- A local simulator keeps the first milestone reproducible and easier to debug.
- Once the generated labels look reasonable, training a CNN should be straightforward.

Initial implementation plan:
1. Create a small forest-fire simulator.
2. Generate grids across several initial tree densities.
3. Label samples using the final burned fraction.
4. Save arrays and metadata in a model-ready format.
5. Create example plots to inspect class quality before training.

Implementation notes:
- Added `src/forest_fire_sim.py`, a NumPy cellular automaton with four states: empty, tree, burning, and burned.
- Added `scripts/generate_dataset.py` to save generated grids and metadata.
- Added `scripts/plot_examples.py` to inspect examples by label.

Initial smoke test:
- A 60-sample run completed, but the original critical-label band was too narrow.
- Most samples either barely spread or burned nearly the full connected tree cluster.
- Adjusted labels to:
  - `subcritical`: burned fraction `< 0.10`
  - `critical`: burned fraction `0.10` to `< 0.90`
  - `supercritical`: burned fraction `>= 0.90`
- Added more density values near the expected 2D percolation transition region.

Second smoke test:
- Command:
  - `python scripts/generate_dataset.py --samples 120 --size 64 --output-dir data/smoke_test`
- Result:
  - `subcritical`: 35
  - `critical`: 51
  - `supercritical`: 34
- Generated inspection figure:
  - `outputs/smoke_test_examples.png`
- Visual check:
  - Subcritical examples are mostly lower-density grids that barely spread.
  - Critical examples appear around the middle density range and have partial burn outcomes.
  - Supercritical examples are dense grids with almost complete burn outcomes.
  - This is balanced enough to proceed to a baseline CNN after creating the full dataset.
