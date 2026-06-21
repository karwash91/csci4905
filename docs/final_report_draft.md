# Synthetic Forest Fire Criticality Detection

## Introduction

This project explores whether a neural network can detect critical behavior in a synthetic forest-fire system. The main idea is to simulate many forest grids, run a simple fire-spread process, label each sample by how much of the forest eventually burns, and train a convolutional neural network to classify the result as subcritical, critical, or supercritical.

The project is inspired by self-organized criticality and phase-transition problems. Near a critical threshold, small local changes can produce much larger system-level effects. In this project, the threshold is represented by whether a fire dies out quickly, spreads through part of the forest, or burns almost the entire connected tree structure.

## Data Generation

I implemented a local NumPy cellular automaton instead of relying on an external simulator. Each grid cell has one of four states:

| Value | Meaning |
| ---: | --- |
| 0 | empty ground |
| 1 | tree |
| 2 | burning tree |
| 3 | burned tree |

Each simulation starts by sampling a random tree grid at a chosen density. A few occupied cells are ignited, then fire spreads through adjacent trees using a 4-neighbor rule. The simulation stops when no burning cells remain.

Each sample is labeled by final burned fraction:

| Label | Burned Fraction |
| --- | --- |
| subcritical | `< 0.10` |
| critical | `0.10` to `< 0.90` |
| supercritical | `>= 0.90` |

The final dataset contains 3000 generated samples at `64x64` resolution.

Class distribution:

| Class | Count |
| --- | ---: |
| subcritical | 1003 |
| critical | 1144 |
| supercritical | 853 |

Example grid figure:

`outputs/synthetic_examples.png`

## Model

The baseline model is a small convolutional neural network. The input is a one-channel `64x64` grid normalized from cell states `0..3` into the range `0..1`.

Architecture summary:

- 3 convolution blocks
- ReLU activations
- Max pooling in the first two blocks
- Adaptive average pooling
- Small fully connected classifier

Training setup:

| Setting | Value |
| --- | --- |
| loss | cross entropy |
| optimizer | Adam |
| learning rate | 0.001 |
| batch size | 64 |
| epochs | 15 |
| split | stratified 70/15/15 |

## Results

The CNN achieved:

| Metric | Value |
| --- | ---: |
| best validation accuracy | 0.9196 |
| test accuracy | 0.8940 |
| test loss | 0.2380 |

Confusion matrix:

| Actual \ Predicted | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 139 | 12 | 0 |
| critical | 18 | 140 | 15 |
| supercritical | 0 | 3 | 126 |

Training curves:

`outputs/baseline_cnn/training_curves.png`

Confusion matrix figure:

`outputs/baseline_cnn/confusion_matrix.png`

## Analysis

The baseline CNN performed well enough to validate the synthetic-data workflow. It learned to separate the two extreme cases reliably: subcritical fires almost never get confused with supercritical fires.

Most errors occur in the critical class. This makes sense because the critical class represents the transition region, where samples can resemble either nearby extreme depending on the exact tree connectivity. The model appears to learn a useful density/connectivity signal rather than simply memorizing one class.

## Feature Space Analysis

To better understand what the CNN learned internally, I extracted the 64-dimensional feature vector immediately before the classifier head and projected those features into two dimensions using PCA.

Feature-space output:

`outputs/feature_space/cnn_feature_pca.png`

PCA explained variance:

| Component | Explained Variance |
| --- | ---: |
| PC1 | 0.9166 |
| PC2 | 0.0481 |

Class centroids in PCA space:

| Class | PC1 | PC2 |
| --- | ---: | ---: |
| subcritical | -7.4876 | -0.5159 |
| critical | 0.7966 | 1.0712 |
| supercritical | 7.7359 | -0.8299 |

The PCA plot shows that the learned representation is strongly ordered by criticality. Subcritical samples cluster on one side, supercritical samples cluster on the other, and critical samples mostly sit between them. This supports the idea that the CNN learned a meaningful transition-related representation rather than only memorizing labels.

The critical class still overlaps with nearby regions, which matches the confusion matrix. This is expected because critical samples are boundary cases between fires that die out and fires that spread across most of the map.

## Limitations

The simulator is intentionally simple. It does not model wind, terrain, moisture, regrowth, variable ignition probability, or real satellite imagery. The labels are based on a practical burned-fraction threshold rather than a mathematically exact critical point.

The model is also a baseline. It uses only the initial grid state and predicts the eventual criticality class. More advanced approaches could include time-series frames, richer simulation parameters, or a dedicated autoencoder/VAE trained directly on the grid states.

## Next Steps

Potential extensions:

- Train on larger `128x128` grids.
- Add stochastic spread probabilities.
- Include time-series snapshots from the simulation.
- Train an autoencoder or VAE to compare its latent clustering against the CNN feature-space PCA.
- Compare the local simulator against `gym_forestfire` or another established forest-fire model.
