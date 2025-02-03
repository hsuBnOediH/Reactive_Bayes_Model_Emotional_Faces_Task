# README

## Overview
This repository contains various scripts for getting familiar with REactive Bayes. The primary script of interest, `HGF_emotional_faces.jl`, implements an inference model using the Hierarchical Gaussian Filter (HGF) to analyze participants' behavior on the Emotional Faces task.

## `HGF_emotional_faces.jl`
This script models behavioral responses in the Emotional Faces task using a probabilistic generative model. We attempt to infer parameters that describe participants' behavior, particularly learning rates and decision biases, by fitting an HGF to observed data.

### Summary of the Code
- The script sets up the Julia environment, installing and importing necessary packages.
- It defines the sigmoid function and establishes a probabilistic model for learning parameters (κ, ω, and β) using a structured mean-field variational approach.
- The model includes hierarchical updates, with a higher-layer hidden state influencing the lower-layer state through a Gaussian random walk.
- Observations (task responses) are modeled as Bernoulli-distributed with a probability determined by a sigmoid transformation.
- The script initializes priors, sets inference constraints, and runs variational inference with 20 iterations.
- Results, including inferred hidden states and parameter posteriors, are visualized and saved as images.

### Output
- Estimated hidden states plots (`EF_model_results_smoothing.png`)
- Bethe free energy plot (`Bethe_free_energy.png`)
- Marginal posterior distributions of κ and ω (`EF_marginal_posteriors.png`)

This script provides a structured way to infer latent cognitive parameters from behavioral data using reactive Bayesian inference in Julia.
