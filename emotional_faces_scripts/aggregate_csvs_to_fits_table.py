import sys
import os
import tempfile
import pandas as pd
import neptune

# Parse Neptune run ID from command line
run_id = sys.argv[1]

# Read Neptune API token
with open('/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/carter_neptune_token.txt', 'r') as f:
    token = f.read().strip()

# Connect to existing Neptune run
run = neptune.init_run(
    project="navid.faryad/Emotional-Faces-RxInfer",
    api_token=token,
    with_id=run_id,
)

print(f"Connected to Neptune run: {run_id}")

results_dir = run["batch/results_directory"].fetch()
print(f"Using results directory: {results_dir}")


rows = []
for f in os.listdir(results_dir):
    if f.startswith("model_results_") and f.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(results_dir, f))
            rows.append(df)
        except Exception as e:
            print(f"Error {f}: {e}")

# Log to Neptune
for row in rows:
    run["fits_table"].append(row.to_dict())

print(f"Logged {len(rows)} rows to fits_table.")


# Get the summary statistics
combined = pd.concat(rows, ignore_index=True)
# Compute averages, skipping NaNs
avg_free_energy = combined["free_energy"].mean(skipna=True)
avg_action_prob = combined["mean_action_prob"].mean(skipna=True)
avg_model_acc = combined["mean_model_accuracy"].mean(skipna=True)

num_fits_with_free_energy = combined["free_energy"].notna().sum()
num_fits_with_action_prob = combined["mean_action_prob"].notna().sum()

run["avg_free_energy"] = avg_free_energy
run["avg_action_prob"] = avg_action_prob
run["avg_model_accuracy"] = avg_model_acc
run["num_fits_with_free_energy"] = num_fits_with_free_energy
run["num_fits_with_action_prob"] = num_fits_with_action_prob

run.stop()