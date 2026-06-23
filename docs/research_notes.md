# Research Notes

## Question

Would a larger `128x128` grid make the final project stronger?

Short answer: it is worth testing, but it probably will not solve the main weakness by itself. A larger grid gives more spatial resolution and reduces finite-size noise, but the current simulator still makes density the dominant predictor. The stronger experiment is to combine larger grids with a density-controlled or near-threshold dataset.

## Relevant Literature

### Forest-fire criticality

Clar, Drossel, and Schwabl review the self-organized critical forest-fire model and emphasize that these models are meant to study systems that move toward critical behavior without manually tuning initial conditions. This supports the broader framing of the project, but it also shows that a full SOC model is more dynamic than our current one-shot burn simulator.

Source:
- Siegfried Clar, Barbara Drossel, Franz Schwabl, "Forest fires and other examples of self-organized criticality"
- https://arxiv.org/abs/cond-mat/9610201

### Finite-size effects

Schenk, Drossel, Clar, and Schwabl study finite-size effects in self-organized critical forest-fire models. Their result is useful for our project because it shows that grid size matters, but not in a simple "bigger is automatically better" way. They report that smaller systems can become more homogeneous and have large density fluctuations, while larger systems show patch structure.

Source:
- Klaus Schenk, Barbara Drossel, Siegfried Clar, Franz Schwabl, "Finite-size effects in the self-organized critical forest-fire model"
- https://arxiv.org/abs/cond-mat/9904356

### Percolation threshold

Newman and Ziff give a high-precision square-lattice site percolation threshold of approximately `0.59274621`. This matters because our simulator uses a square grid with 4-neighbor spread. If the project wants to study criticality more directly, densities near this threshold should be emphasized.

Source:
- M. E. J. Newman, R. M. Ziff, "Efficient Monte Carlo algorithm and high-precision results for percolation"
- https://arxiv.org/abs/cond-mat/0005264

### Machine learning phase transitions

Carrasquilla and Melko show that neural networks can learn phases and phase transitions from raw simulated configurations. They also report that performance improves with larger system size in an Ising-model experiment, which supports testing larger grids. The warning for our project is that models can learn simple order parameters when those explain the labels.

Source:
- Juan Carrasquilla, Roger G. Melko, "Machine learning phases of matter"
- https://arxiv.org/abs/1605.01735

Zhang, Liu, and Wei apply machine learning to percolation and related phase transitions. Their paper is especially relevant because it directly connects percolation, 2D lattices, machine learning, and cross-size generalization.

Source:
- Wanzhou Zhang, Jiayu Liu, Tzu-Chieh Wei, "Machine learning of phase transitions in the percolation and XY models"
- https://arxiv.org/abs/1804.02709

## Interpretation for This Project

Increasing from `64x64` to `128x128` should make individual simulations less noisy and may sharpen the transition region. However, because our labels are mostly driven by global tree density, a larger grid may make the density shortcut even cleaner. That could improve accuracy without proving the CNN learned spatial criticality.

The better research-supported direction is:

1. Run the same CNN and ablations on `128x128`.
2. Compare normal CNN accuracy, density-only accuracy, and shuffled-grid accuracy.
3. Add a near-threshold dataset centered around density `0.5927`.
4. Add stochastic spread probability so two samples with similar density can have different outcomes.
5. Use a density-controlled split or report per-density-bin accuracy.

## Recommended Next Experiment

Generate a `128x128` dataset and repeat the existing model plus ablations:

```bash
python scripts/generate_dataset.py --samples 3000 --size 128 --output-dir data/synthetic_128
python scripts/train_cnn.py --data-dir data/synthetic_128 --output-dir outputs/baseline_cnn_128 --epochs 15 --batch-size 64
python scripts/run_ablations.py --data-dir data/synthetic_128 --model-path outputs/baseline_cnn_128/model.pt --output-dir outputs/ablations_128
```

Decision rule:

- If `128x128` improves CNN accuracy and the shuffled-grid result drops, larger grids helped the model use spatial layout.
- If `128x128` improves CNN accuracy but shuffled-grid performance stays close, the model is still mostly density-driven.
- If density-only accuracy increases too, the dataset became easier because density is cleaner at larger grid size.

## Stronger Follow-Up

The strongest follow-up is a near-threshold experiment:

- Sample more densities around `0.5927`, such as `0.54` to `0.65`.
- Add random spread probability, for example `0.70` to `1.00`.
- Compare models within narrow density bins.
- Report whether the CNN beats density-only thresholds inside the near-critical region.

That would better answer whether the model sees connectivity patterns, not just density.
