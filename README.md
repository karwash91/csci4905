# Synthetic Forest Fire Generation and Criticality Detection

This project simulates forest-fire spread, trains a conditional VAE to generate new forest layouts, and trains neural models to classify outcomes as `subcritical`, `critical`, or `supercritical`.

1. A conditional VAE generates new forest layouts for requested criticality classes.
2. A baseline CNN predicts the broad dataset well.
3. Shortcut tests show the broad dataset has a density shortcut.
4. A fixed-density dataset removes that shortcut and makes spatial layout matter.
5. A burn-mask U-Net recovers many critical examples by predicting the burned region first.
6. Connectivity baselines explain the physical mechanism: reachable connected tree structure is the real signal.

Generated data, plots, and model weights are ignored by Git.
The `docs/` directory is also local-only so reports, slides, notes, and reference PDFs can be submitted separately without cluttering the code repository.

## Submission Contents

| Path | Purpose |
| --- | --- |
| `README.md` | project overview and reproduction commands |
| `src/forest_fire_sim.py` | forest-fire simulator |
| `scripts/` | data generation, model training, ablations, and slide builder |

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

## Reproduce Main Results

Generate and train on the broad-density dataset:

```bash
python scripts/generate_dataset.py --samples 3000 --size 64 --output-dir data/synthetic
python scripts/plot_examples.py --data-dir data/synthetic --output outputs/synthetic_examples.png
python scripts/train_conditional_vae.py --data-dir data/synthetic --output-dir outputs/conditional_vae --epochs 18 --batch-size 64 --latent-dim 32 --beta 0.05 --generated-per-class 40
python scripts/train_cnn.py --data-dir data/synthetic --output-dir outputs/baseline_cnn --epochs 15 --batch-size 64
python scripts/train_multitask_cnn.py --data-dir data/synthetic --output-dir outputs/multitask_cnn_w05 --epochs 40 --batch-size 64 --burn-loss-weight 0.5 --patience 15 --scheduler-patience 5 --scheduler-factor 0.5
python scripts/evaluate_burn_thresholds.py --data-dir data/synthetic --model-path outputs/multitask_cnn_w05/model.pt --output-dir outputs/burn_threshold_eval
python scripts/run_ablations.py --data-dir data/synthetic --model-path outputs/baseline_cnn/model.pt --output-dir outputs/ablations
python scripts/connectivity_baseline.py --data-dir data/synthetic --output-dir outputs/connectivity_baseline
```

Generate and train on the fixed-density spatial challenge:

```bash
python scripts/generate_spatial_dataset.py --samples-per-class 500 --size 64 --density 0.55 --output-dir data/spatial_64
python scripts/plot_examples.py --data-dir data/spatial_64 --output outputs/spatial_examples.png
python scripts/train_cnn.py --data-dir data/spatial_64 --output-dir outputs/spatial_cnn --epochs 30 --batch-size 64 --patience 8 --scheduler-patience 4 --scheduler-factor 0.5
python scripts/run_ablations.py --data-dir data/spatial_64 --model-path outputs/spatial_cnn/model.pt --output-dir outputs/spatial_ablations
python scripts/train_burn_mask_unet.py --data-dir data/spatial_64 --output-dir outputs/spatial_burn_mask_unet --epochs 25 --batch-size 32 --patience 6
python scripts/connectivity_baseline.py --data-dir data/spatial_64 --output-dir outputs/spatial_connectivity_baseline
```

## Main Results

Broad-density dataset:

| Generative Check | Requested Samples | Simulator Match Rate |
| --- | ---: | ---: |
| conditional VAE generated layouts | 120 | 0.9917 |

Conditional VAE generated-sample matrix:

| Requested \ Simulated | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 40 | 0 | 0 |
| critical | 1 | 39 | 0 |
| supercritical | 0 | 0 | 40 |

| Method | Accuracy | Critical Recall |
| --- | ---: | ---: |
| baseline CNN | 0.8940 | 0.809 |
| multi-task classifier head | 0.8962 | 0.850 |
| burn prediction with tuned thresholds | 0.9007 | 0.873 |

Broad-density ablation:

| Check | Accuracy | Critical Recall |
| --- | ---: | ---: |
| density-only thresholds | 0.8764 | 0.832 |
| CNN on normal grids | 0.8940 | 0.809 |
| CNN on shuffled grids | 0.8896 | 0.803 |

Fixed-density spatial challenge:

| Check | Accuracy | Critical Recall |
| --- | ---: | ---: |
| density-only thresholds | 0.3333 | 0.000 |
| CNN on normal grids | 0.6667 | 0.000 |
| CNN on shuffled grids | 0.3333 | 0.000 |
| burn-mask U-Net | 0.6800 | 0.747 |

Connectivity baselines:

| Dataset | Baseline | Accuracy | Critical Recall |
| --- | --- | ---: | ---: |
| broad-density | density-only | 0.8764 | 0.832 |
| broad-density | largest connected component | 0.9294 | 0.925 |
| broad-density | ignition-reachable component | 1.0000 | 1.000 |
| fixed-density | density-only | 0.3333 | 0.000 |
| fixed-density | largest connected component | 0.7111 | 0.213 |
| fixed-density | ignition-reachable component | 1.0000 | 1.000 |

## Local Docs

These files are useful for submission/presentation, but ignored by Git:

- `docs/final_report_draft.md` - simplified final report
- `docs/final_project_slide_deck.md` - presentation slide outline
- `docs/final_project_narration_script.md` - speaker notes
- `docs/research_notes.md` - literature notes and future experiment ideas

## Key Outputs

These files are generated locally and ignored by Git:

- `outputs/synthetic_examples.png`
- `outputs/conditional_vae/generated_examples.png`
- `outputs/conditional_vae/generated_match_matrix.png`
- `outputs/spatial_examples.png`
- `outputs/baseline_cnn/confusion_matrix.png`
- `outputs/multitask_cnn_w05/confusion_matrix.png`
- `outputs/spatial_cnn/confusion_matrix.png`
- `outputs/spatial_burn_mask_unet/confusion_matrix.png`
- `outputs/burn_threshold_eval/burn_threshold_eval.md`
- `outputs/ablations/ablation_results.md`
- `outputs/connectivity_baseline/connectivity_baseline.md`
- `outputs/spatial_connectivity_baseline/connectivity_baseline.md`
