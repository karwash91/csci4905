# Synthetic Forest Fire Generation and Criticality Detection

## Introduction

This project studies whether a neural network can generate and evaluate simple forest-fire examples. I generated many forest grids, started a few fires, let the fire spread, and then labeled each example based on how much of the forest burned.

The three labels are:

- `subcritical`: the fire burns only a small part of the forest
- `critical`: the fire burns a medium amount
- `supercritical`: the fire burns almost everything connected to it

The generative part of the project uses a conditional VAE to create new forest layouts for a requested class. I then run the simulator on those generated grids to check whether they actually behave like the requested class.

The main lesson is that generation and classification both need to be tested carefully. The conditional VAE can generate examples that match the requested class on the broad dataset, but that dataset also has a tree-density shortcut. After I controlled density, the layout of the trees became much more important. The best neural approach for the layout task was not just predicting the class directly, but first predicting which cells would burn.

This project was also a good example of using an AI coding assistant productively. I used Codex to help brainstorm experiments, write and revise scripts, debug results, and turn the project into a clearer report. The important part was not just asking for code, but using the assistant to ask better questions about what the model was actually learning.

## Simulation

I built a simple forest-fire simulator with NumPy. Each grid cell has one of four values:

| Value | Meaning |
| ---: | --- |
| 0 | empty ground |
| 1 | tree |
| 2 | burning tree |
| 3 | burned tree |

Each simulation starts with a random forest and a few burning trees. At each step, fire spreads to neighboring trees up, down, left, or right. The simulation stops when no trees are burning.

Each example is labeled by final burned fraction:

| Label | Burned Fraction |
| --- | --- |
| subcritical | `< 0.10` |
| critical | `0.10` to `< 0.90` |
| supercritical | `>= 0.90` |

The first dataset contains 3000 generated `64x64` grids:

| Class | Count |
| --- | ---: |
| subcritical | 1003 |
| critical | 1144 |
| supercritical | 853 |

Example output: `outputs/synthetic_examples.png`

## Conditional VAE Generator

To make the project clearly generative, I trained a conditional variational autoencoder, or conditional VAE. This is similar to the VAE idea from Project 2, but with one extra input: the requested fire-spread class.

The model learns to generate a forest layout for one of these requested labels:

- `subcritical`
- `critical`
- `supercritical`

The model outputs a probability map for where trees should appear. I sample a new forest grid from that probability map, add a few starting fires, and run the same simulator used for the training data. This lets me check whether the generated forest actually produces the requested class.

Training command:

`python scripts/train_conditional_vae.py --data-dir data/synthetic --output-dir outputs/conditional_vae --epochs 18 --batch-size 64 --latent-dim 32 --beta 0.05 --generated-per-class 40`

Generated-sample check:

| Requested Class | Generated Samples | Simulator Match Rate | Mean Burned Fraction |
| --- | ---: | ---: | ---: |
| subcritical | 40 | 1.000 | 0.019 |
| critical | 40 | 0.975 | 0.480 |
| supercritical | 40 | 1.000 | 0.982 |
| overall | 120 | 0.992 | n/a |

Generated-sample matrix:

| Requested \ Simulated | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 40 | 0 | 0 |
| critical | 1 | 39 | 0 |
| supercritical | 0 | 0 | 40 |

Example output: `outputs/conditional_vae/generated_examples.png`

This shows that the conditional VAE can generate forest grids that usually behave like the requested class after simulation. However, this result should be read carefully. On the broad dataset, class is strongly related to tree density, so the VAE is partly learning how dense each class should look. That is still a valid generative result, but it also motivates the later shortcut tests.

## Baseline CNN

The first model was a small convolutional neural network. It takes the initial forest grid as input and predicts one of the three labels.

The baseline CNN performed well:

| Metric | Value |
| --- | ---: |
| test accuracy | 0.8940 |
| test loss | 0.2380 |
| critical recall | 0.809 |

Here, `critical recall` means: out of all examples that were actually critical, how many did the model correctly find?

Confusion matrix:

| Actual \ Predicted | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 139 | 12 | 0 |
| critical | 18 | 140 | 15 |
| supercritical | 0 | 3 | 126 |

This result showed that the CNN could predict the fire outcome from the starting grid. However, accuracy alone did not show what the CNN was using to make its decision.

At this point, Codex helped suggest a useful follow-up question: was the CNN learning tree layout, or was it mostly using a simpler shortcut like tree density? That led to the next set of tests.

## Adding Burned Fraction Prediction

The class labels come from burned fraction, so I also trained a model to predict burned fraction directly. This is similar to Project 1, where changing the output and tracking better metrics helped explain what the model was learning.

The model had two outputs:

- the three-class prediction
- the predicted burned fraction

The best neural result came from using the predicted burned fraction and then applying class cutoffs to it.

| Method | Accuracy | Critical Recall |
| --- | ---: | ---: |
| baseline CNN | 0.8940 | 0.809 |
| multi-output CNN class prediction | 0.8962 | 0.850 |
| burned-fraction prediction with tuned cutoffs | 0.9007 | 0.873 |

The tuned cutoffs were `0.1100` and `0.8914`, which are close to the original cutoffs of `0.10` and `0.90`. This suggests the model learned a useful estimate of burned fraction, not just a random extra output.

## Testing For A Density Shortcut

Next, I wanted to know whether the CNN was really using tree layout or mostly using tree density. Tree density means the percentage of grid cells that contain trees.

I tested this in two ways:

- Density-only test: predict using only tree density.
- Shuffled-grid test: keep the same number of trees, but randomly move them around.

| Check | Accuracy | Critical Recall |
| --- | ---: | ---: |
| density-only prediction | 0.8764 | 0.832 |
| CNN on normal grids | 0.8940 | 0.809 |
| CNN on shuffled grids | 0.8896 | 0.803 |

The shuffled-grid CNN was almost as accurate as the normal CNN. This means the first dataset was mostly driven by density. The CNN was still learning something useful, but it did not need much information about where the trees were placed.

This changed the direction of the project. The original task was too easy because density alone explained a lot of the answer.

This was one of the most useful places for AI assistance. Instead of only trying to increase accuracy, Codex helped suggest checking for shortcuts with a density-only model and a shuffled-grid test. Those tests made the project stronger because they showed a weakness in the first dataset.

## Fixed-Density Layout Dataset

To make layout matter more, I generated a second dataset where every grid has the same density: `0.55`. Since density is fixed, the model cannot use density as the main shortcut.

Instead, the dataset changes how the trees are arranged:

- random layout
- clustered layout
- fragmented layout

This dataset is balanced:

| Class | Count |
| --- | ---: |
| subcritical | 500 |
| critical | 500 |
| supercritical | 500 |

Example output: `outputs/spatial_examples.png`

The results changed:

| Check | Accuracy | Critical Recall |
| --- | ---: | ---: |
| density-only prediction | 0.3333 | 0.000 |
| CNN on normal grids | 0.6667 | 0.000 |
| CNN on shuffled grids | 0.3333 | 0.000 |

Since there are three classes, chance accuracy is about `0.3333`. The density-only method dropped to chance, as expected. The CNN did better on normal grids, but went back to chance when the grids were shuffled. This shows that tree layout mattered in the fixed-density dataset.

However, the direct CNN still missed the critical class. It learned the easier difference between low-burn and high-burn examples, but it did not handle the middle cases well.

The fixed-density dataset also came from that same back-and-forth process. Codex helped suggest controlling density so the model would have to use layout instead of relying on the number of trees.

## Predicting The Burned Area

A human would probably not solve this by guessing the class immediately. A human would trace which trees are connected to the fire, estimate what area burns, and then decide the class.

To make the model closer to that process, I trained a small image-to-image CNN to predict the burned area first. This type of model is often called a U-Net, but the important idea is simple: instead of outputting one label, it outputs a whole grid. The input has two channels:

- where the trees are
- where the fire starts

The output is a predicted burn mask. A `mask` is just a grid that marks which cells belong to something. In this case, it marks which cells the model thinks will burn.

After the model predicts the burn mask, I calculate the predicted burned fraction and convert that into the three labels.

On the fixed-density dataset, this was the strongest neural result for layout:

| Model | Accuracy | Critical Recall | Mask Overlap | Pixel Accuracy |
| --- | ---: | ---: | ---: | ---: |
| direct CNN classifier | 0.6667 | 0.000 | n/a | n/a |
| burn-mask U-Net | 0.6800 | 0.747 | 0.651 | 0.941 |

`Mask overlap` measures how much the predicted burned area overlaps with the true burned area. Higher is better.

Burn-mask confusion matrix:

| Actual \ Predicted | subcritical | critical | supercritical |
| --- | ---: | ---: | ---: |
| subcritical | 34 | 41 | 0 |
| critical | 10 | 56 | 9 |
| supercritical | 0 | 12 | 63 |

The burn-mask model found `56/75` critical examples. The direct CNN found none. Its total accuracy only improved a little because it sometimes called subcritical examples critical, but it was much better at finding the middle class.

This was the clearest neural-network evidence that layout can be learned.

This experiment was another place where Codex was useful. After the direct classifier missed the critical class, Codex suggested thinking about how a human would solve the problem: trace the burn area first, then classify it. That led to the burn-mask model.

## Connectivity Baseline

Finally, I compared the neural models to simple hand-made features based on connectivity.

A connected component is a group of trees that touch each other through up, down, left, or right neighbors. This matters because fire can only spread through connected trees.

I tested three simple baselines:

- density only
- largest connected group of trees
- trees connected to the starting fire

The last one is not a fair learned model. It is more like an answer key for this simple simulator, because the fire always spreads by the same neighbor rule. I included it to show what physical rule the neural networks are trying to learn.

| Dataset | Baseline | Accuracy | Critical Recall |
| --- | --- | ---: | ---: |
| first dataset | density-only | 0.8764 | 0.832 |
| first dataset | largest connected group | 0.9294 | 0.925 |
| first dataset | connected to starting fire | 1.0000 | 1.000 |
| fixed-density | density-only | 0.3333 | 0.000 |
| fixed-density | largest connected group | 0.7111 | 0.213 |
| fixed-density | connected to starting fire | 1.0000 | 1.000 |

These results explain the project clearly. Density is useful, but connected tree structure is more important. The neural models learn part of that structure, while the hand-made connectivity rule captures it directly.

Codex also helped suggest this simple connectivity baseline. This was helpful because it gave an easy-to-understand comparison against the neural networks and showed why the burn-mask model made sense.

## Discussion

The project followed a clear path.

First, I added a generative model. The conditional VAE generated new forest layouts and `119/120` generated samples simulated into the requested class.

Second, the baseline CNN looked successful with `0.8940` test accuracy. The burned-fraction model improved this to `0.9007`.

Third, the shortcut tests showed that this success was not the full story. On the first dataset, density explained most of the result.

Fourth, the fixed-density dataset removed that shortcut. In that dataset, layout mattered. The direct CNN still missed the middle class, but the burn-mask model found many critical examples because it learned a more useful intermediate task.

The main takeaway is that the target matters. A generative model can create useful examples, but it can also reflect shortcuts in the data. Directly predicting a class can also lead the model to use shortcuts. Predicting burned fraction or the burned area gives the model information that better matches the real fire process.

Overall, this class project was very AI-assistant friendly because there were many small decisions to make: what model to try, what baseline to compare against, how to test for shortcuts, and how to explain the results. Codex helped with those steps, but the final report still depends on running the experiments and checking whether the results actually support the story.

## Limitations

The simulator is simple. It does not include wind, terrain, moisture, regrowth, random fire spread, or real satellite images. The labels are practical cutoffs based on burned fraction, not exact scientific critical points.

The connectivity baseline is also not a normal machine-learning model. It uses knowledge of how the simulator works. I used it to explain the physical pattern behind the results.

## Next Steps

Possible next steps:

- Add random spread probability so connected trees do not always burn.
- Train the conditional VAE on the fixed-density layout dataset.
- Train on larger `128x128` grids.
- Predict fire spread one step at a time.
- Test examples near density `0.5927`, where large connected tree groups start to appear on square grids.
- Add wind, terrain, or moisture.

## Conclusion

The conditional VAE added a real generative AI component by creating new forest layouts for requested fire-spread classes. The first CNN result was also good, but it mostly used density. After controlling density, tree layout mattered. The direct classifier still struggled with the critical middle class, but the burn-mask model recovered many of those examples by predicting the burned area first.

The strongest conclusion is that the model should match the process being modeled. For this forest-fire task, generation was useful, but evaluation mattered just as much. Predicting an intermediate spatial result was more useful than predicting the final class directly.
