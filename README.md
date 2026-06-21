# Synthetic Forest Fire Criticality Detection

This final project generates synthetic forest-fire simulations, labels each sample by fire-spread behavior, and trains neural networks to detect criticality.

The final story is:

1. Build a simple cellular automaton forest-fire simulator.
2. Generate a balanced synthetic dataset.
3. Train a baseline CNN classifier.
4. Show that longer classification-only training plateaus quickly.
5. Improve the project story with a multi-task CNN that predicts both criticality class and burned fraction.
6. Analyze the learned CNN feature space with PCA.

## Project Structure

- `src/forest_fire_sim.py` - NumPy forest-fire cellular automaton.
- `scripts/generate_dataset.py` - generate synthetic grids and metadata.
- `scripts/plot_examples.py` - plot labeled simulation examples.
- `scripts/train_cnn.py` - baseline classification CNN.
- `scripts/train_multitask_cnn.py` - CNN with classification and burned-fraction regression heads.
- `scripts/analyze_latent_space.py` - PCA analysis of CNN feature vectors.
- `scripts/summarize_results.py` - baseline metrics summary.
- `scripts/compare_model_results.py` - compare baseline and multi-task runs.
- `docs/project_log.md` - running project log with decisions, commands, and results.
- `docs/final_report_draft.md` - report draft.

Generated data, plots, and model weights are intentionally ignored by Git.

## Setup

Install the needed packages:

```bash
pip install numpy matplotlib torch
```

## Reproduce The Main Run

Generate the dataset:

```bash
python scripts/generate_dataset.py --samples 3000 --size 64 --output-dir data/synthetic
```

Create example plots:

```bash
python scripts/plot_examples.py --data-dir data/synthetic --output outputs/synthetic_examples.png
```

Train the baseline classifier:

```bash
python scripts/train_cnn.py \
  --data-dir data/synthetic \
  --output-dir outputs/baseline_cnn \
  --epochs 15 \
  --batch-size 64
```

Train the multi-task model:

```bash
python scripts/train_multitask_cnn.py \
  --data-dir data/synthetic \
  --output-dir outputs/multitask_cnn_w05 \
  --epochs 40 \
  --batch-size 64 \
  --burn-loss-weight 0.5 \
  --patience 15 \
  --scheduler-patience 5 \
  --scheduler-factor 0.5
```

Run feature-space analysis:

```bash
python scripts/analyze_latent_space.py \
  --data-dir data/synthetic \
  --model-path outputs/baseline_cnn/model.pt \
  --output-dir outputs/feature_space
```

Compare model results:

```bash
python scripts/compare_model_results.py --output outputs/model_comparison.md
```

## Current Results

Dataset:

| Class | Count |
| --- | ---: |
| subcritical | 1003 |
| critical | 1144 |
| supercritical | 853 |

Model comparison:

| Run | Test Acc | Burn MAE | Critical Recall | Supercritical Recall |
| --- | ---: | ---: | ---: | ---: |
| baseline CNN | 0.8940 | n/a | 0.809 | 0.977 |
| multi-task CNN, weight 0.1 | 0.8940 | 0.0547 | 0.844 | 0.938 |
| multi-task CNN, weight 0.5 | 0.8962 | 0.0554 | 0.850 | 0.938 |
| multi-task CNN, weight 1.0 | 0.8962 | 0.0558 | 0.850 | 0.938 |

The baseline CNN is already strong, and longer classification-only training plateaued quickly. The multi-task CNN gives the best project story because it improves recall on the critical transition class and also predicts continuous burned fraction with low error.

## Key Outputs

- `outputs/synthetic_examples.png`
- `outputs/baseline_cnn/training_curves.png`
- `outputs/baseline_cnn/confusion_matrix.png`
- `outputs/multitask_cnn_w05/training_curves.png`
- `outputs/multitask_cnn_w05/confusion_matrix.png`
- `outputs/feature_space/cnn_feature_pca.png`
- `outputs/model_comparison.md`

