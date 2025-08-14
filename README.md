# Investigating Human Inference: Message Passing vs Sampling

This repository contains code for investigating whether humans use **message passing** or **sampling** to perform inference. The comparison is performed by fitting both models to **human choices and reaction times** from the *Emotional Faces Task*.

---

## 🧠 Task Summary

In the **Emotional Faces Task**, participants are presented with:

1. A **low** or **high tone** noise  
2. Followed immediately by an image of either an **angry** or **sad** face  

Key details:
- The tone and face occur in quick succession
- The image is shown **very briefly** (0.15 seconds)
- Participants respond with the **left** or **right** arrow key to indicate whether they saw an **angry** or **sad** face
- If no response is made quickly enough, `"too slow"` is displayed before the next trial

The objective is to determine whether human inference in this task is better explained by a **message passing model** or a **sampling-based model**.

---


---

## 📁 pyddm_scripts

This folder contains the **drift-diffusion model (DDM)** implementation and fitting code using [`pyddm`](https://github.com/mwshinn/pyddm).

### `fit_pyddm_emotional_faces.py`
Fits the DDM to behavioral data from the Emotional Faces Task:
- Loads processed data for a given subject
- Defines a custom loss function (`EF_Loss`) which calls the model in `DDM_model.py`
- Fits the model to empirical data to estimate drift and learning parameters
- Simulates behavior from fitted parameters and re-fits to check recoverability
- Saves parameter estimates, model fit statistics, and simulated outputs

### `DDM_model.py`
Implements the trial-by-trial Kalman filter-based DDM for the task:
- Computes drift values based on trial type (`angry` vs `sad`)
- Updates expected value estimates using a learning rate
- Fits: calculates reaction time PDFs, choice probabilities, prediction errors, and absolute RT errors
- Simulations: generates synthetic choices and RTs given model parameters
- Returns detailed trial-by-trial model statistics, including entropy measures

---

## 📁 RxInfer_scripts

This folder contains the **message passing model** of the task implemented using [RxInfer](https://biaslab.github.io/RxInfer.jl/) and **reactive message passing**.  
The model uses probabilistic graphical modeling to represent the task and infers latent variables via message passing rather than sampling.

---

## ⚙️ Dependencies

Python scripts require:
- Python 3.8+
- [`pyddm`](https://github.com/mwshinn/pyddm)
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

RxInfer scripts require:
- Julia 1.8+
- [`RxInfer.jl`](https://biaslab.github.io/RxInfer.jl/)

