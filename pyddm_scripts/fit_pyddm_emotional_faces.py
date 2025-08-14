import pyddm
import argparse
from pyddm import Fitted 
import pandas as pd
import numpy as np
from DDM_model import DDM_model
from scipy.io import savemat
import sys, random
import pyddm.plot
import matplotlib.pyplot as plt
import pickle # only needed for debugging purposes when a model_stats object is loaded into the loss function
from pyddm import BoundConstant, Fitted, BoundCollapsingLinear # USed for debugging purposes when fixing parameters in the loss function

# Either retreive passed in arguments or use defaults
parser = argparse.ArgumentParser()
parser.add_argument('--subject_id', type=str, default='5b915c9d22d85d000115c0f5')
parser.add_argument('--results_dir', type=str, default=f"L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/pyddm_test/")
parser.add_argument('--root', type=str, default='L:/')
args = parser.parse_args()

subject_id = args.subject_id
results_dir = args.results_dir
root = args.root

# Get timestamp
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

log_path = f"{results_dir}{subject_id}_{timestamp}_model_log.txt"
sys.stdout = open(log_path, "w", buffering=1)  # line-buffered so that it updates the file in real-time
sys.stderr = sys.stdout  # Also capture errors
print("Welcome to the pyddm model fitting script!")

# Set the random seed for reproducibility
seed = 23
np.random.seed(seed)
random.seed(seed)
print(f"Random seed set to {seed}")
eps = np.finfo(float).eps

########### Load in emotional faces data and format as Sample object ###########
with open(f"{root}rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/prolific_emotional_faces_processed_data_07-16-25-17_42_14/task_data_{subject_id}_processed_data.csv", "r") as f:
    raw = pd.read_csv(f)

df = raw.copy()
# Replace -200 with "undecided" in resp_response column
df["resp_response"] = df["resp_response"].replace("-200", "undecided")

settings = dict()
settings["df"] = df

df_mapped = df.copy()
df_mapped["resp_response"] = df_mapped["resp_response"].map({"left": 0, "right": 1})

df_dropped= df_mapped.copy()
df_dropped = df_mapped.loc[df_mapped["resp_response"].notna()]

emotional_faces_sample = pyddm.Sample.from_pandas_dataframe(df_dropped, rt_column_name="resp_response_time", choice_column_name="resp_response", choice_names=("right", "left")) # note the ordering here is intentional since pyddm codes the first choice as 1 (upper) and the second as 0 (lower) which matches our coding left as 0 and right as 1


class EF_Loss(pyddm.LossFunction):
    name = "EF_Loss"
    def loss(self, model):

        model_stats = DDM_model(model.settings["df"],model, fit_or_sim="fit")

        # get log likelihood
        rt_pdf = model_stats["rt_pdf"]
        action_probs = model_stats["action_probs"]
        abs_error_RT = model_stats["abs_error_RT"]
        eps = np.finfo(float).eps
        likelihood = np.sum(np.log(rt_pdf[~np.isnan(rt_pdf)] + eps)) # add small value to avoid log(0)
        avg_abs_error_RT = np.mean((abs_error_RT[~np.isnan(abs_error_RT)]))


        # Store model statistics
        model.action_probs = action_probs
        model.rt_pdf = rt_pdf
        model.abs_error_RT = abs_error_RT
        model.avg_abs_error_RT = avg_abs_error_RT
        model.model_acc = model_stats["model_acc"]
        model.exp_vals = model_stats["exp_vals"]
        model.pred_errors = model_stats["pred_errors"]
        model.alpha = model_stats["alpha"]

        return -likelihood





## FIT MODEL TO ACTUAL DATA
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
print("Setting up the model to fit behavioral data")
model_to_fit = pyddm.gddm(drift=lambda drift_mod, learning_rate, drift_value : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,    bound=3,
                          nondecision=0, T_dur=7,
                          conditions=["drift_value","starting_position_value"],
                          parameters={"drift_mod": (-1, 1), "learning_rate": (0,1)}, choice_names=("right","left"))
model_to_fit.settings = settings


print("Fitting behavioral data")
model_to_fit.fit(sample=emotional_faces_sample, lossfunction=EF_Loss,verbose=True)

# Save fit parameter estimates
# model_to_fit.get_fit_result()
params = model_to_fit.parameters()
# Extract the fitted parameters into a flat dictionary for easier access
fit_result = {}
for subdict in params.values():
    for name, val in subdict.items():
        if isinstance(val, Fitted):
            fit_result[f"post_{name}"] = float(val)
            fit_result[f"min_{name}"] = val.minval
            fit_result[f"max_{name}"] = val.maxval
        else:
            fit_result[f"fixed_{name}"] = val

# Extract the model accuracy and average action probability
fit_result["average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=emotional_faces_sample, lossfunction=EF_Loss)
# Store the number of invalid trials
fit_result["avg_abs_error_RT"] = model_to_fit.avg_abs_error_RT

# Store the model output
model_output = {
    "action_probs": model_to_fit.action_probs,
    "rt_pdf": model_to_fit.rt_pdf,
    "abs_error_RT": model_to_fit.abs_error_RT,
    "predicted_RT": model_to_fit.predicted_RT,
    "rt_pdf_entropy": model_to_fit.rt_pdf_entropy,
    "model_acc": model_to_fit.model_acc,
    "exp_vals": model_to_fit.exp_vals,
    "pred_errors": model_to_fit.pred_errors,
    "alpha": model_to_fit.alpha,
}


# Save the fitted model before it gets written over during recoverability analysis
model_fit_to_data = model_to_fit
# Examine recoverability on fit parameters
# This is kind of hacky, but we pass in the learning parameters as arguments to drift even though they aren't used for it. This is just so the loss function has access to them
print("Setting up the model to simulate behavioral data")
model_to_sim = pyddm.gddm(drift=lambda drift_mod, learning_rate, drift_value : drift_value,
                          starting_position=lambda starting_position_value: starting_position_value, 
                          noise=1.0,    bound=3,
                          nondecision=0, T_dur=7,
                          conditions=["drift_value","starting_position_value"],
                          parameters={
    "drift_mod": fit_result["post_drift_mod"],
    "learning_rate": fit_result["post_learning_rate"],
}, choice_names=("right","left"))
model_to_sim.settings = settings


print("Simulating behavioral data")
simulated_behavior = DDM_model(df,model_to_sim,fit_or_sim="sim")

# Assign simulated behavior to df field in the model
model_to_fit.settings["df"] = simulated_behavior["data"]


# Fit the simulated data
print("Fitting simulated behavioral data")
model_to_fit.fit(sample=emotional_faces_sample, lossfunction=EF_Loss)
params = model_to_fit.parameters()
model_fit_to_simulated_data = model_to_fit
# Extract the parameter estimates for the model fit to the simulated data (hence prefix sft)
simfit_result = {}
for subdict in params.values():
    for name, val in subdict.items():
        if isinstance(val, Fitted):
            simfit_result[f"sft_post_{name}"] = float(val)
            simfit_result[f"sft_min_{name}"] = val.minval
            simfit_result[f"sft_max_{name}"] = val.maxval
        else:
            simfit_result[f"sft_fixed_{name}"] = val
fit_result.update(simfit_result)

fit_result["sft_average_action_prob"] = np.nanmean(model_to_fit.action_probs)
fit_result["sft_model_acc"] = np.nanmean(model_to_fit.model_acc)
fit_result["sft_final_loss"] = pyddm.get_model_loss(model=model_to_fit, sample=emotional_faces_sample, lossfunction=EF_Loss)



print("Finished saving results. Exiting script!")
