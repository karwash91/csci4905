Final Project Manifest: Synthetic Forest Fire Criticality Detection

Introduction and Goals

This project aims to generate synthetic forest fire images using a cellular automaton simulation, label them according to criticality thresholds, and train a neural network to detect when the system is near a critical phase transition. It draws inspiration from research on criticality in dynamical systems and generative models . The focus is on exploring emergent behaviours and latent representations in dynamical systems using generative AI and computer vision.

External Resources and References

* Gym Forest Fire (sahandrez/gym_forestfire): an open source simulation of the Drossel and Schwabl forest-fire model built as a Gym environment. The environment represents a forest as a 128×128 grid with cell states for empty ground, trees, and burning trees, and it defines rules for ignition, propagation and regrowth

. This will be used to generate synthetic training and test images.

* Self-Organized Criticality Simulation (SocSim): a Python project containing models of self-organized criticality, including the forest-fire model and sandpile models . It offers alternative simulation code and insights into SOC behaviour.
* Research on Neural Networks and Criticality: the paper Smallest Neural Network to Learn the Ising Criticality (Kim & Kim, 2018) shows that small neural networks can learn to identify critical behaviour in the Ising model. Additional papers on generative models for phase transitions and deep learning for wildfire prediction provide context for modeling and evaluation.

High-Level Plan and Workflow (approx. 12 hours)

1. Environment Setup (1 hour)

* Access the school’s GPU server and ensure necessary permissions.
* Install Python dependencies: PyTorch, numpy, gym, matplotlib, and any additional packages required by gym_forestfire or SocSim.
* Clone the gym_forestfire repository and verify that you can run the simulation examples

.

2. Simulation and Data Generation (3 hours)

* Explore the gym_forestfire environment to understand its parameters and output format. Run several short simulations to observe how the forest evolves over time at different growth and lightning probabilities.
* Write a script to generate a large number of simulations. For each simulation, save the initial grid and label each instance according to:
    * Tree density (sparse, moderate, dense).
    * Whether the fire eventually spreads across the map (subcritical, critical, supercritical).
* Optionally use SocSim to cross‑validate the behaviour of the synthetic data or to generate alternative grid patterns .
* Save grids as images or NumPy arrays along with their labels in a format ready for model training.

3. Data Preprocessing and Labeling (1 hour)

* Convert raw grid arrays into consistent image tensors. If using grayscale images, map states to intensity values (e.g. empty=0, tree=1, burning=2) and normalize.
* Verify the class distribution and generate more samples if certain categories are underrepresented. Adjust simulation parameters to balance classes.
* Split the dataset into training, validation, and test sets, ensuring that each set includes a range of densities and criticality states.

4. Model Selection and Design (1 hour)

* Start with a simple convolutional neural network (CNN) classifier to predict criticality categories from images. Design a small architecture with a few convolutional layers, followed by fully connected layers and softmax output.
* Consider training a variational autoencoder (VAE) or masked autoencoder (MAE) to learn latent representations of the grids; use the encoder’s latent vectors for clustering and analysis.
* Determine loss functions and training hyperparameters (learning rate, batch size, epochs). Implement early stopping and checkpoint saving to fit within the available time.

5. Model Training (3 hours)

* Train the CNN on the training set using the GPU server. Monitor accuracy and loss on the validation set, adjusting hyperparameters if training stagnates or overfits.
* If time allows, train a VAE/MAE on the same data. Track reconstruction loss and latent consistency; use the encoder output for later analysis.
* Save the best performing model weights for later evaluation.

6. Model Evaluation and Analysis (2 hours)

* Evaluate the trained classifier on the test set. Compute metrics such as accuracy, precision, recall and confusion matrices to assess performance across criticality classes.
* For autoencoder models, project latent space vectors using dimensionality reduction (e.g. t‑SNE) and identify clustering patterns. Determine whether images near the critical threshold cluster in a distinct region of the latent space.
* Compare results against insights from relevant literature and note any patterns that align with or diverge from theoretical expectations.

7. Documentation and Reporting (1 hour)

* Draft a report or presentation summarizing the methodology, simulation settings, model architectures, training results and analysis. Include visual examples of synthetic grids, learning curves and latent space plots.
* Document any deviations from the initial plan, explaining what changes were made and why.
* Provide references and links to the external projects and research papers used during the project.

Flexibility and Adaptation

This plan is intended as a guideline and may be adjusted during execution based on findings or constraints. If data generation or model training takes longer than expected, prioritize completing a baseline classification model and report. Optional extensions such as training generative models or exploring cross-domain transfer can be pursued only if time permits.