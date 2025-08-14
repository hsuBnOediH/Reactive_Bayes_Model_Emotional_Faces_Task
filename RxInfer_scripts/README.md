# README

## Overview
This repository contains various scripts for getting familiar with REactive Bayes. The primary script of interest, `HGF_emotional_faces.jl`, implements an inference model using the Hierarchical Gaussian Filter (HGF) to analyze participants' behavior on the Emotional Faces task.

There are two workflows for running the scripts: locally and on a Linux cluster.

## Running Locally
When running locally, the Project.toml and Manifest.toml files are located in the same directory as the script. To set up the environment, simply open the Julia REPL in the project directory and activate the environment:
import Pkg
Pkg.activate(".")
Pkg.instantiate()

## Running on a Linux Cluster
On the Linux cluster, use the provided HGF_loop.py script to submit jobs via SLURM. This script calls HGF_emotional_faces.jl after setting up the environment. When running on the cluster, the script activates the project in a dedicated cluster environment directory:

### Handling Tree Has Mismatch Errors
A tree hash mismatch error was encountered when adding certain packages (e.g., Plots, RxInfer). This error occurred because the package artifacts were being downloaded to a network file system, leading to file inconsistencies. The solution was to download dependencies locally by setting the Julia Depot Path. To do this, add the following commands to your startup script (or SLURM job script):
export JULIA_DEPOT_PATH="/var/tmp/$USER/julia_depot_shared"
mkdir -p "$JULIA_DEPOT_PATH"
export JULIA_PKG_PRECOMPILE_AUTO=0
This configuration directs Julia to store and precompile packages on a local disk, thus avoiding network-related issues.

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
- Hidden states and parameter plots (e.g., hidden_states_and_parameters_<subject>_<datetime>.png)
- Free energy convergence plot (e.g., free_energy<subject>_<datetime>.png)
- Processed results saved as CSV (e.g., model_results_<subject>_<datetime>.csv)

This script provides a structured way to infer latent cognitive parameters from behavioral data using reactive Bayesian inference in Julia.
