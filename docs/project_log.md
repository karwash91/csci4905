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

### 2026-06-21 - Baseline CNN Classifier

Dataset generation:
- Command:
  - `python scripts/generate_dataset.py --samples 3000 --size 64 --output-dir data/synthetic`
- Result:
  - `subcritical`: 1003
  - `critical`: 1144
  - `supercritical`: 853
- Generated inspection figure:
  - `outputs/synthetic_examples.png`

Model:
- Added `scripts/train_cnn.py`.
- Input: one-channel `64x64` grid normalized from cell states `0..3` to `0..1`.
- Architecture: small CNN with three convolution blocks, adaptive average pooling, and a small fully connected classifier.
- Loss: cross entropy.
- Optimizer: Adam with learning rate `0.001`.
- Split: stratified `70/15/15`.

Training command:
- `python scripts/train_cnn.py --data-dir data/synthetic --output-dir outputs/baseline_cnn --epochs 15 --batch-size 64`

Training result:
- Dataset size: 3000
- Split sizes:
  - train: 2099
  - validation: 448
  - test: 453
- Best validation accuracy: 0.9196
- Test accuracy: 0.8940
- Test loss: 0.2380
- Saved outputs:
  - `outputs/baseline_cnn/model.pt`
  - `outputs/baseline_cnn/metrics.json`
  - `outputs/baseline_cnn/training_curves.png`
  - `outputs/baseline_cnn/confusion_matrix.png`

Confusion matrix:

| Actual \\ Predicted | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 139 | 12 | 0 |
| critical | 18 | 140 | 15 |
| supercritical | 0 | 3 | 126 |

Interpretation:
- The baseline CNN performs well enough to validate the synthetic-data workflow.
- Most errors involve the `critical` class, which is expected because it represents the transition region rather than a clean extreme.
- The model almost never confuses subcritical directly with supercritical, suggesting it learned a meaningful density/connectivity signal.

### 2026-06-21 - Results Summary and Report Draft

Added:
- `scripts/summarize_results.py`
- `docs/final_report_draft.md`

Summary command:
- `python scripts/summarize_results.py --metrics outputs/baseline_cnn/metrics.json --output outputs/baseline_cnn/results_summary.md`

Per-class test metrics:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| subcritical | 0.885 | 0.921 | 0.903 | 151 |
| critical | 0.903 | 0.809 | 0.854 | 173 |
| supercritical | 0.894 | 0.977 | 0.933 | 129 |

Interpretation:
- The `critical` class has the weakest recall because it is the ambiguous transition class.
- The model is strongest on `supercritical` recall, correctly identifying nearly all high-spread simulations.
- The report draft now has sections for introduction, data generation, model, results, analysis, limitations, and next steps.

### 2026-06-21 - CNN Feature-Space Analysis

Added:
- `scripts/analyze_latent_space.py`

Command:
- `python scripts/analyze_latent_space.py --data-dir data/synthetic --model-path outputs/baseline_cnn/model.pt --output-dir outputs/feature_space`

Method:
- Loaded the trained baseline CNN.
- Extracted the 64-dimensional feature vector before the classifier head for all 3000 samples.
- Projected the feature vectors into two dimensions using PCA.
- Plotted the embedding colored by criticality label.

Results:
- Feature shape: `(3000, 64)`
- PCA explained variance:
  - PC1: 0.9166
  - PC2: 0.0481
- Class centroids:
  - subcritical: `[-7.4876, -0.5159]`
  - critical: `[0.7966, 1.0712]`
  - supercritical: `[7.7359, -0.8299]`
- Saved outputs:
  - `outputs/feature_space/cnn_feature_pca.png`
  - `outputs/feature_space/feature_space_summary.json`

Interpretation:
- The CNN feature space is strongly ordered by criticality along PC1.
- Critical samples mostly sit between subcritical and supercritical samples.
- This supports the claim that the classifier learned a meaningful transition-related representation.
- Some overlap around the critical region is expected because those samples represent boundary behavior.

### 2026-06-21 - Longer Classification Run Plateau

Goal:
- Compare whether longer training budgets were likely to improve the baseline classifier before spending 30, 60, or 90 minutes on longer runs.

Change:
- Updated `scripts/train_cnn.py` to support:
  - validation-loss checkpointing
  - wall-clock time limits with `--max-minutes`
  - early stopping with `--patience`
  - `ReduceLROnPlateau`
  - saved per-epoch history in `metrics.json`

Trial command:
- `python -u scripts/train_cnn.py --data-dir data/synthetic --output-dir outputs/refine_30m --epochs 1000 --batch-size 64 --max-minutes 30 --patience 30 --min-delta 0.0005 --scheduler-patience 8 --scheduler-factor 0.5`

Observed state before stopping:
- Validation loss improved quickly early, then plateaued.
- Earlier baseline best validation loss was around `0.2148` near epoch 12.
- Longer run reached only about `0.2109` by epoch 53.
- Validation accuracy stayed mostly around `0.917` to `0.922`.
- Learning rate reductions did not produce meaningful additional gains.

Decision:
- Stop extending the same classification-only CNN.
- The likely bottleneck is the coarse class label, especially for the `critical` transition class.
- Next experiment: train a multi-task model that predicts both criticality class and continuous burned fraction.

Reasoning:
- The class labels are thresholded from burned fraction.
- Two samples labeled `critical` can have very different burned fractions, such as `0.11` and `0.88`.
- A regression head gives the model smoother information about proximity to the transition instead of only the coarse class bucket.

### 2026-06-21 - Multi-Task CNN Refinement

Added:
- `scripts/train_multitask_cnn.py`

Model:
- Shared CNN feature extractor.
- Classification head predicts:
  - `subcritical`
  - `critical`
  - `supercritical`
- Regression head predicts continuous burned fraction in `[0, 1]`.

Loss:
- `total loss = cross entropy class loss + burn_loss_weight * SmoothL1 burned-fraction loss`
- Tested `burn_loss_weight` values: `0.1`, `0.5`, and `1.0`.

Recommended command:
- `python -u scripts/train_multitask_cnn.py --data-dir data/synthetic --output-dir outputs/multitask_cnn_w05 --epochs 40 --batch-size 64 --burn-loss-weight 0.5 --patience 15 --scheduler-patience 5 --scheduler-factor 0.5`

Results:
- Best epoch by validation loss: 31
- Test accuracy: 0.8962
- Test loss: 0.2325
- Test burned-fraction MAE: 0.0554
- Test burned-fraction RMSE: 0.0956
- Saved outputs:
  - `outputs/multitask_cnn_w05/model.pt`
  - `outputs/multitask_cnn_w05/metrics.json`
  - `outputs/multitask_cnn_w05/training_curves.png`
  - `outputs/multitask_cnn_w05/confusion_matrix.png`

Multi-task confusion matrix:

| Actual \\ Predicted | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 138 | 13 | 0 |
| critical | 15 | 147 | 11 |
| supercritical | 0 | 8 | 121 |

Comparison with baseline CNN:

| Run | Test Acc | Test Loss | Burn MAE | Critical Recall | Supercritical Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline CNN | 0.8940 | 0.2380 | n/a | 0.809 | 0.977 |
| multi-task, weight 0.1 | 0.8940 | 0.2305 | 0.0547 | 0.844 | 0.938 |
| multi-task, weight 0.5 | 0.8962 | 0.2325 | 0.0554 | 0.850 | 0.938 |
| multi-task, weight 1.0 | 0.8962 | 0.2349 | 0.0558 | 0.850 | 0.938 |

Interpretation:
- Overall accuracy improved only slightly.
- Critical recall improved, which is valuable because critical behavior is the main target of the project.
- The improvement came with a tradeoff: supercritical recall dropped slightly.
- The burned-fraction head is useful for explaining proximity to the transition, even if it does not dramatically improve classification accuracy.
- The `0.5` and `1.0` settings produced the same classification confusion matrix; `0.5` is the preferred final setting because it has slightly lower total loss and burned-fraction error.

### 2026-06-21 - Project Story Simplification

Added:
- `scripts/compare_model_results.py`

Command:
- `python scripts/compare_model_results.py --output outputs/model_comparison.md`

Documentation update:
- Replaced the original planning manifest in `README.md` with the actual project workflow and current results.
- Final story is now:
  1. Simulate synthetic forest-fire grids.
  2. Label by burned fraction.
  3. Train a baseline CNN.
  4. Show that longer classification-only training plateaus.
  5. Add a burned-fraction regression head to improve critical-region recall.
  6. Use CNN feature-space PCA to show learned ordering by criticality.
