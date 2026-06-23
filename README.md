# Synthetic Forest Fire Generation and Criticality Detection

This project simulates forest-fire spread, trains a conditional VAE to generate new forest layouts, and trains neural models to classify outcomes as `subcritical`, `critical`, or `supercritical`.

1. A conditional VAE generates new forest layouts for requested criticality classes.
2. A baseline CNN predicts the broad dataset well.
3. Shortcut tests show the broad dataset has a density shortcut.
4. A fixed-density dataset removes that shortcut and makes spatial layout matter.
5. A burn-mask U-Net recovers many critical examples by predicting the burned region first.
6. Connectivity baselines explain the physical mechanism: reachable connected tree structure is the real signal.

## Setup

```bash
pip install numpy matplotlib torch
```

## Core Scripts

| Script | Purpose |
| --- | --- |
| `src/forest_fire_sim.py` | NumPy cellular automaton simulator |
| `scripts/generate_dataset.py` | broad-density dataset |
| `scripts/generate_spatial_dataset.py` | fixed-density spatial challenge |
| `scripts/train_conditional_vae.py` | conditional VAE forest-layout generator |
| `scripts/train_cnn.py` | direct CNN classifier |
| `scripts/train_multitask_cnn.py` | classifier plus burned-fraction regression |
| `scripts/evaluate_burn_thresholds.py` | threshold burned-fraction predictions |
| `scripts/run_ablations.py` | density-only and shuffled-grid checks |
| `scripts/train_burn_mask_unet.py` | burn-mask segmentation model |
| `scripts/connectivity_baseline.py` | connected-component baselines |
| `scripts/plot_examples.py` | example grid figures |
| `scripts/build_slide_deck.py` | rebuild the presentation deck |
