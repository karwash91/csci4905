# Final Project Narration Script

This is written as a natural speaker script for the current 14-slide deck. It should fit roughly 8 to 10 minutes if read at a normal pace.

## Slide 1: Generative AI for Supercritical Forest-Fire Detection

For my final project, I looked at a simulated forest-fire system and asked whether a model could detect when a fire becomes supercritical.

In plain language, supercritical means the fire does not just burn a small local area. It spreads through most of what it can reach.

The project combines three ideas from the course: simulation, neural networks, and generative AI. I generated synthetic forest grids, trained models to predict fire outcomes, and also tested whether a generative model could create new forest layouts with a requested outcome.

## Slide 2: Research Question

The main question is: can a model look at the starting forest and predict whether the fire will become supercritical?

The image shows examples from the simulator. Each grid starts as a forest with some trees and initial fire locations. After the simulation runs, the example is labeled by how much of the forest burns.

The important class for this presentation is the supercritical class. That is the high-risk case, where the fire reaches almost everything connected to it.

## Slide 3: Project Pipeline

This slide shows the full project pipeline.

First, I generate simulated forest-fire examples and measure the actual burn outcome. Then I train a generative model, the conditional VAE, to create new forest layouts for a requested class.

The important step is that generated grids are not accepted just because they look reasonable. I run them back through the simulator to check whether they actually behave like the requested class.

After that, I train detection models to predict whether a starting grid will be subcritical, critical, or supercritical. Finally, I run stress tests, like density-only prediction and shuffled grids, to see whether the model learned the intended pattern or just found a shortcut.

## Slide 4: Background: Criticality

This project connects back to Project 3, where I worked with forest density and image generation.

The background idea is criticality. Some systems sit near a tipping point. Below that point, a small event stays small. Near or above that point, the same small event can spread widely.

For this project, the tipping behavior is represented by burned fraction. If less than 10 percent burns, I call it subcritical. If 10 to 90 percent burns, I call it critical. If 90 percent or more burns, I call it supercritical.

So the project is really about predicting the big-burn case before the fire actually spreads.

## Slide 5: Simulation Setup

I used a simple NumPy simulator so that the result would be measurable and repeatable.

Each cell in the grid can be empty, tree, burning, or burned. The fire spreads only up, down, left, and right. It does not spread diagonally.

The simulation stops when there are no burning trees left. Then I calculate the burned fraction and assign the class label.

This setup is simpler than a real wildfire, but that is useful for a class project because I can control the data and know exactly why each label was assigned.

## Slide 6: Generative Add-On

Because this is a generative AI class, I added a conditional VAE. The goal was not to make realistic forest photos. The goal was to generate new synthetic forest grids for a requested class.

For example, I could ask the VAE to generate a supercritical layout. Then I would run that generated grid through the simulator and check whether it actually behaved as supercritical.

The matrix shows that 119 out of 120 generated samples matched the requested class after simulation. All requested supercritical samples matched.

This connects to Project 2. With generated outputs, it is not enough to say the images look right. I need a quantitative check. Here, the simulator gives that check.

The honest limitation is that this generative step is useful for synthetic grids, but it does not mean the model is generating realistic wildfire images.

## Slide 7: Baseline Supercritical Detector

The first prediction model was a basic CNN. It looked at the starting grid and predicted subcritical, critical, or supercritical.

The initial result looked strong. Test accuracy was about 89 percent, and supercritical recall was 0.977. Recall means that out of the true supercritical examples, the model found almost all of them.

On this confusion matrix, the bottom-right cell is the key one for supercritical detection. The model correctly identified 126 out of 129 supercritical examples.

But Project 1 taught me not to stop at accuracy. A model can have good accuracy and still be using an easy shortcut.

## Slide 8: Density Shortcut Test

So the next question was: is the CNN really learning tree layout, or is it mostly learning density?

Density means how many cells contain trees. If dense forests usually burn more, then a model might not need much spatial reasoning.

I tested this in two ways. First, I made a density-only baseline. Second, I shuffled the test grids so the number of trees stayed the same, but their positions changed.

The result was important. The density-only method was almost as strong as the CNN, and the shuffled CNN still performed almost as well as the normal CNN.

That showed that the first dataset had a density shortcut. The model was useful, but the task was easier than I originally thought.

## Slide 9: Fixed-Density Challenge

To remove that shortcut, I created a fixed-density dataset.

In this dataset, every forest has density 0.55. That means each grid has the same number of trees. The only major difference is where those trees are placed.

Now the density-only method drops to chance, which is what I wanted. The CNN still performs above chance on normal grids, but when the grids are shuffled, performance drops back down.

This is the evidence that layout can matter. Once density is controlled, where the trees are placed affects the outcome.

The direct CNN still had a weakness, though. It was good at separating the easy low-burn and high-burn cases, but it missed the middle critical class.

## Slide 10: Burn-Mask U-Net

At this point, I changed the learning target.

Instead of asking the model to immediately guess the class, I asked it to predict which cells would burn. This is closer to how a person would solve the problem.

A person would trace the trees connected to the starting fire, estimate the burned area, and then decide if the result is subcritical, critical, or supercritical.

The burn-mask model is a small U-Net. Its input has two channels: one channel shows where the trees are, and the other channel shows where the fire starts.

The output is not one class label. The output is a burn mask, which is a grid showing which cells the model thinks will burn. Then I convert that predicted mask into burned fraction and class label.

## Slide 11: Burn-Mask Results

This slide shows what happened when I used the burn-mask model on the fixed-density dataset.

This model still detected supercritical examples well, with 63 out of 75 correct. More importantly, it recovered many critical examples that the direct CNN missed.

So the strongest result here is not just higher accuracy. It is that predicting an intermediate spatial outcome helped the model behave more like the actual fire process.

## Slide 12: Final Takeaways

This chart summarizes the main story.

The baseline CNN was already strong at supercritical detection, but the density shortcut test showed that the original dataset was partly too easy.

The fixed-density dataset made layout matter. Then the burn-mask model showed that spatial reasoning is more achievable when the target matches the process.

The main lesson from earlier projects carried through here. From Project 1, I used more than accuracy. From Project 2, I checked generated outputs quantitatively. From Project 3, I paid attention to forest density and how strongly it controls the result.

My final takeaway is that the model design should match the system being modeled. For this task, predicting the burned area first is more meaningful than jumping straight to a class label.

## Slide 13: Resources

This slide lists the main resources behind the project.

On the left are the project resources: the simulator, the conditional VAE script, the CNN and U-Net scripts, the final report, and the research notes. Those are the files that make the project reproducible.

On the right are the research ideas that shaped the project. The forest-fire criticality papers helped frame the system. The percolation paper helped explain why density around a threshold matters. The machine-learning phase-transition papers helped justify using neural networks on simulated grid systems, while also warning that models can learn simple shortcuts.

I also included Codex because it helped with brainstorming, debugging, and organizing experiments, especially around the shortcut tests.

## Slide 14: Glossary

This final slide is mainly here as a reference for the technical terms.

The most important terms are CNN, VAE, U-Net, recall, and supercritical.

A CNN is the basic image-style model I used for classification. A VAE is the generative model used to create new synthetic grids. A U-Net is the model that predicts the burned mask. Recall measures how many true examples of a class the model correctly finds.

And supercritical is the main risk class in this project: the case where at least 90 percent of the reachable forest burns.

That is the overall project path: generate simulated forests, detect supercritical fires, discover the density shortcut, control for it, and then use a better spatial target to predict what burns.
