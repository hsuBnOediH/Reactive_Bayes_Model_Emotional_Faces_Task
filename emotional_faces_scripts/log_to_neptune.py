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
with open('carter_neptune_token.txt', 'r') as file:
    neptune_api_token = file.read().strip()

# Initialize Neptune run under the same batch run ID
run = neptune.init_run(
    project="navid.faryad/Emotional-Faces-RxInfer",
    api_token={neptune_api_token},
    with_id=batch_run_id,
)  
# Append free energy and action prob for current fit
run["metrics/free_energy"].append(df.loc[0, "free_energy_last"])
run["metrics/action_prob"].append(df.loc[0, "mean_action_prob"])
run.wait()
# print(run["metrics/free_energy"].fetch_values())

# Fetch previous values. Convert to float because they're retrieved as dataframe
free_energy_df = run["metrics/free_energy"].fetch_values()
action_prob_df = run["metrics/action_prob"].fetch_values()
# Convert to float (use .tolist() to extract values)
prev_free_energy_vals = free_energy_df["value"].astype(float).tolist()
prev_action_probs = action_prob_df["value"].astype(float).tolist()

# Compute averages
run["avg_free_energy"] = sum(prev_free_energy_vals) / len(prev_free_energy_vals)
run["avg_action_prob"] = sum(prev_action_probs) / len(prev_action_probs)
run["number_of_fits"] = len(prev_action_probs)

# Log subject-specific metrics inside the batch
subject_run = run[f"fits/{subject_id}"]

# Log metrics
subject_run["metrics/kappa_mean"] = df.loc[0, "kappa_mean"]
subject_run["metrics/kappa_std"] = df.loc[0, "kappa_std"]
subject_run["metrics/omega_mean"] = df.loc[0, "omega_mean"]
subject_run["metrics/omega_std"] = df.loc[0, "omega_std"]
subject_run["metrics/beta_mean"] = df.loc[0, "beta_mean"]
subject_run["metrics/beta_std"] = df.loc[0, "beta_std"]
subject_run["metrics/free_energy_last"] = df.loc[0, "free_energy_last"]
subject_run["metrics/mean_action_prob"] = df.loc[0, "mean_action_prob"]

# Upload CSV file to Neptune
subject_run["results_csv"].upload(csv_path)

print(f"Neptune logging complete for batch {batch_run_id}, subject {subject_id}")

run.stop()
