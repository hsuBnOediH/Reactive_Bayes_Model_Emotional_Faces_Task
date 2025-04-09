import sys, os, re, subprocess, neptune
from datetime import datetime 
current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

results = f"/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_{current_datetime}"
if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")


ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/run_HGF_rxinfer.ssub'

stdout_name = f"{results}/logs/EF-%J.stdout"
stderr_name = f"{results}/logs/EF-%J.stderr"

predictions_or_responses = "responses"

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/emotional_faces_prolific_IDs.csv'
subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        subjects.append(line.strip())

# Read in Neptune API Token
with open('/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/carter_neptune_token.txt', 'r') as file:
    neptune_api_token = file.read().strip()

# sys.exit("Stopping early for debugging")

# Initialize Neptune run to log hyperparameters and results
run = neptune.init_run(
    project="navid.faryad/Emotional-Faces-RxInfer",
    api_token=neptune_api_token,
)  
run["batch/cluster"] = "true"
run["batch/results_directory"] = results
# Fetch the actual run ID so I can append results of fits to this run
batch_run_id = run["sys/id"].fetch()  
print(f"Batch Run ID: {batch_run_id}")

# Define hyperparameters
hyperparameters = {
    "prior_kappa_mean" : 1.0,
    "prior_kappa_variance" : 0.1,
    "prior_omega_mean" : 0.0,
    "prior_omega_variance" : .01, # used to be .01
    "prior_beta_shape" : 1.0,
    "prior_beta_rate" : 1.0,
    "prior_z_mean" : 0.0,
    "prior_z_variance" : 1.0,
    "prior_x_mean" : 0.0,
    "prior_x_variance" : 1.0,
    "initial_z_mean" : 0.0,
    "initial_z_variance" : 10.0,
    "initial_kappa_mean" : 1.0,
    "initial_kappa_variance" : 0.1,
    "initial_omega_mean" : 0.0,
    "initial_omega_variance" : 0.01,
    "initial_beta_shape" : 0.1,
    "initial_beta_rate" : 0.1,
    "niterations" : 20,
}
# Log hyperparameters to Neptune
for key, value in hyperparameters.items():
    run[f"hyperparameters/{key}"] = value

# Log source code in Neptune
run["source_code/files"].upload_files([
    "/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/HGF_loop.py",
    "/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/run_HGF_rxinfer.ssub",
    "/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/HGF_emotional_faces.jl"
])
# Create hyperaparameter string to pass into bash
hyperparam_str = ",".join(f"{key}={value}" for key, value in hyperparameters.items())
print(f"Hyperparams String: {hyperparam_str}")  # Debugging

# Fit each subjects' data
for subject in subjects:
    # if subject != "62bf234d622ba94e5dfdb997" and subject != "5ac53209fa3b4e0001736f22"   and subject != "5afa19a4f856320001cf920f": 
    #     continue
    jobname = f'{batch_run_id}-RxInfer-{subject}'

    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results} {subject} {predictions_or_responses} {batch_run_id} {hyperparam_str}")

    print(f"SUBMITTED JOB [{jobname}]")


# Call the script to wait and log all the results
agg_ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/wait_and_log.ssub'
agg_stdout = f"{results}/logs/wait_and_log-%J.stdout"
agg_stderr = f"{results}/logs/wait_and_log-%J.stderr"
agg_jobname = f'EF-RxInfer-WaitAndLog'
os.system(f"sbatch -J {agg_jobname} -o {agg_stdout} -e {agg_stderr} {agg_ssub_path} {batch_run_id}")



#python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/HGF_loop.py






