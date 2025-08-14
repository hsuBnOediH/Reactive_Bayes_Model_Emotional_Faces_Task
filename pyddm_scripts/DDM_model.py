# KF DDM model file
# Define a function that returns the likelihood of data under a model and can simulate data based on parameters
import pandas as pd
import numpy as np
from pyddm.logger import logger
import matplotlib.pyplot as plt # USed for plot in the model function
from pyddm import BoundConstant, Fitted, BoundCollapsingLinear # USed for debugging purposes when fixing parameters in the loss function
from scipy.stats import norm


# Add a filter to the logger to check for the Renormalization warning, where the probability of hitting the upper, lower, or undecided boundary is not 1 so it had to be renormalized
had_renorm = False
def renorm_filter(record):
    global had_renorm               # ← declare you’re writing the module flag
    if "Renormalizing probability density" in record.getMessage():
        had_renorm = True
    return True

logger.addFilter(renorm_filter)


def DDM_model(data,model,fit_or_sim, sim_using_max_pdf=False):
    global had_renorm 
    eps = np.finfo(float).eps

    num_trials = 200

    drift_mod = model.get_dependence("drift").drift_mod
    learning_rate = model.get_dependence("drift").learning_rate


    rt_pdf = np.full(num_trials, np.nan)
    rt_pdf_entropy = np.full(num_trials, np.nan)
    action_probs = np.full(num_trials, np.nan)
    model_acc = np.full(num_trials, np.nan)
    predicted_RT = np.full(num_trials, np.nan)
    abs_error_RT = np.full(num_trials, np.nan)

    pred_errors = np.full(num_trials, np.nan)
    exp_vals = np.full(num_trials+1, np.nan)
    exp_vals[0] = 0
    alpha = np.full(num_trials, np.nan)



    # --- catch errors during model execution ---
    try:
        for trial_num in range(0, num_trials):
            
            # If saw an angry face
            if "angry" in data["trial_type"].iloc[trial_num]:
                drift_value = -drift_mod
            elif "sad" in data["trial_type"].iloc[trial_num]:
                drift_value = drift_mod

            starting_position_value = np.tanh(exp_vals[trial_num])


            if fit_or_sim == "fit":
                # solve a ddm (i.e., get the probability density function) for current DDM parameters
                # Higher values of reward_diff and side_bias indicate greater preference for right bandit (band it 1 vs 0)
                had_renorm = False

                sol = model.solve(
                    conditions={
                        "drift_value": drift_value,
                        "starting_position_value": starting_position_value,
                    }
                )
                # plt.plot(sol.t_domain, sol.pdf("right"))
                # plt.axvline(x=trial['RT'], color='red', linestyle='--', label='RT')
                # plt.legend()  # Optional, if you want the 'RT' label to show up
                # plt.show()

                # Check to see if the renormalization warning was triggered
                if had_renorm:
                    print("The mode had to renormalize the probability density function. Please check the model parameters:")
                    params = model.parameters()
                    for subdict in params.values():
                        for name, val in subdict.items():
                            print(f"{name} = {float(val)}")
                    print("drift_value = ", drift_value)
                    print("starting_position_value = ", starting_position_value)
                    print()


                # Evaluate the pdf of the reaction time for the chosen option. Note that left will be the bottom boundary and right upper
                # If it was an undecided trial
                if data["resp_response"].iloc[trial_num] == "undecided":
                    pdf = sol.prob_undecided()
                    prob = sol.prob_undecided()
                else:
                    pdf = sol.evaluate(data["resp_response_time"].iloc[trial_num], data["resp_response"].iloc[trial_num])
                    prob = sol.prob(data["resp_response"].iloc[trial_num])
                
                assert pdf >= 0 and prob >=0, "Probability of choice and probability density of a reaction time must be non-negative"
                # Store the probability of the choice under the model
                rt_pdf[trial_num] = pdf
                # Store the action probability of the choice under the model
                action_probs[trial_num] = prob
                model_acc[trial_num] = prob > .5

                # Get the absolute error between the model's predicted reaction time and the actual reaction time
                if max(sol.pdf("left")) > max(sol.pdf("right")):
                    # Get the index of the max pdf
                    predicted_rt_index = int(np.argmax(sol.pdf("left")))
                else:
                    predicted_rt_index = int(np.argmax(sol.pdf("right")))
                predicted_RT[trial_num] = model.dt * predicted_rt_index
                abs_error_RT[trial_num] = abs(data["resp_response_time"].iloc[trial_num] - predicted_RT[trial_num])


            else:
                # Simulate a reaction time from the model based on the parameters

                # solve a ddm (i.e., get the probability density function) for current DDM parameters
                # Higher values of reward_diff and side_bias indicate greater preference for right bandit (band it 1 vs 0)
                sol = model.solve(conditions={
                    "drift_value": drift_value,
                    "starting_position_value": starting_position_value,
                })

                # Use the reaction time with the max probability density
                if sim_using_max_pdf:
                    # If an undecided trial was simulated
                    if sol.prob_undecided() > sol.prob("right") and sol.prob_undecided() > sol.prob("left"):   
                        simmed_choice = "undecided"
                        simmed_rt = pd.NA
                    else:
                        # If the left choice was more likely
                        if max(sol.pdf("left")) > max(sol.pdf("right")):
                            # Get the index of the max pdf
                            rt_index = int(np.argmax(sol.pdf("left")))
                            simmed_choice = "left"
                        else:
                            rt_index = int(np.argmax(sol.pdf("right")))
                            simmed_choice = "right"
                        # Get the reaction time at that index
                        simmed_rt = model.dt * rt_index

                # Sample from the pdf distributions
                else:
                    res = sol.sample(1).to_pandas_dataframe(drop_undecided=True) # We only use the first non-undecided trial
                    # dataframe will be empty if the trial was undecided.
                    if res.empty:
                        # If the dataframe is empty, it means the trial was undecided
                        simmed_choice = "undecided"
                        simmed_rt = pd.NA
                    else:
                        if res.choice[0] == 0:
                            simmed_choice = "left"
                        elif res.choice[0] == 1:
                            simmed_choice = "right"
                        simmed_rt = res.RT[0]

                # Assign the simulated action and RT to the dataframe
                data.loc[(data["trial_number"] == trial_num),"resp_response"] = simmed_choice
                data.loc[(data["trial_number"] == trial_num), "resp_response_time"] = simmed_rt

            # Record the entropy of the reaction time and choice pdf for the trial
            # Compute discrete probability mass
            left_mass = sol.pdf("left") * sol.dt
            right_mass = sol.pdf("right") * sol.dt
            undecided_mass = sol.prob_undecided()
            # Combine entropy terms using safe log
            left_entropy = -np.sum(left_mass * np.log(left_mass + eps))
            right_entropy = -np.sum(right_mass * np.log(right_mass + eps))
            undecided_entropy = -undecided_mass * np.log(undecided_mass + eps)
            # Store total entropy
            rt_pdf_entropy[trial_num] = left_entropy + right_entropy + undecided_entropy


            # If saw an angry face
            if "angry" in data["trial_type"].iloc[trial_num]:
                exp_vals[trial_num+1] = exp_vals[trial_num] + learning_rate*(-1 - exp_vals[trial_num])
            elif "sad" in data["trial_type"].iloc[trial_num]:
                exp_vals[trial_num+1] = exp_vals[trial_num] + learning_rate*(1 - exp_vals[trial_num])


    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Printing the parameter values: ")
        for subdict in model.parameters().values():
            for name, val in subdict.items():
                print(f"  {name} = {val}")
        # Raise the exception so we still get traceback
        raise

    # Clean up dataframe if simulating
    if fit_or_sim == "sim":
        cols_to_keep = [
            "trial_number",
            "trial_type",
            "resp_response_time",
            "resp_response",
            "observed_sad_high_or_angry_low",
            "face_intensity"
        ]

        data = data[cols_to_keep]


    # Return a dictionary of model statistics
    return {
        "action_probs": action_probs,
        "rt_pdf": rt_pdf,
        "model_acc": model_acc,
        "exp_vals": exp_vals,
        "pred_errors": pred_errors,
        "alpha": alpha,
        "predicted_RT": predicted_RT,
        "abs_error_RT": abs_error_RT,
        "rt_pdf_entropy": rt_pdf_entropy, # Entropy of the reaction time and choice pdf
        # "jsd_diff_chosen_opt":jsd_diff_chosen_opt, # JSD difference of the chosen option
        "data": data,
    }
