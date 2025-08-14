import neptune
import sys
import pandas as pd
import os

# Get CSV file and batch_run_id from command-line arguments
csv_path = sys.argv[1]
print(f"CSV path: {csv_path}")
batch_run_id = sys.argv[2]  # This was passed from Bash
print(f"Batch run ID: {batch_run_id}")
# Read the CSV file
df = pd.read_csv(csv_path)

# Extract subject ID from CSV filename
subject_id = os.path.basename(csv_path).split("_")[-1].split(".")[0]

# Read in Neptune API Token
with open('/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/carter_neptune_token.txt', 'r') as file:
    neptune_api_token = file.read().strip()

# Initialize Neptune run under the same batch run ID
run = neptune.init_run(
    project="navid.faryad/Emotional-Faces-RxInfer",
    api_token=neptune_api_token,
    with_id=batch_run_id,
)  
# Append free energy and action prob for current fit
# if not pd.isna(df.loc[0, "free_energy"]):
#     run["metrics/free_energy_list"].append(df.loc[0, "free_energy"])

# run["metrics/action_prob_list"].append(df.loc[0, "mean_action_prob"])
# run.wait()

# # Fetch previous values. Convert to float because they're retrieved as dataframe
# if not pd.isna(df.loc[0, "free_energy"]):
#     free_energy_df = run["metrics/free_energy_list"].fetch_values()
#     prev_free_energy_vals = free_energy_df["value"].astype(float).tolist()
#     # Compute average
#     run["avg_free_energy"] = sum(prev_free_energy_vals) / len(prev_free_energy_vals)
#     run["number_of_fits_with_free_energy"] = len(prev_free_energy_vals)


# action_prob_df = run["metrics/action_prob_list"].fetch_values()
# prev_action_probs = action_prob_df["value"].astype(float).tolist()

# # Compute average
# run["avg_action_prob"] = sum(prev_action_probs) / len(prev_action_probs)
# run["number_of_fits_with_action_prob"] = len(prev_action_probs)


# Log subject-specific metrics inside the batch
# subject_run = run[f"fits/{subject_id}"]

# Be careful -- this code creates race conditions when two jobs upload at the same time
# run["fits_table"].append({
#     "subject_id": subject_id,
#     "free_energy": df.loc[0, "free_energy"],
#     "mean_action_prob": df.loc[0, "mean_action_prob"],
#     "mean_model_accuracy": df.loc[0, "mean_model_accuracy"],
#     "num_iterations_used": df.loc[0, "num_iterations_used"],
#     "kappa_mean": df.loc[0, "kappa_mean"],
#     "kappa_std": df.loc[0, "kappa_std"],
#     "omega_mean": df.loc[0, "omega_mean"],
#     "omega_std": df.loc[0, "omega_std"],
#     "beta_mean": df.loc[0, "beta_mean"],
#     "beta_std": df.loc[0, "beta_std"]
# })


# Upload CSV file to Neptune
run[f"csv_files/{subject_id}"].upload(csv_path)

print(f"Neptune logging complete for batch {batch_run_id}, subject {subject_id}")

run.stop()
