import sys, os, re, subprocess
from datetime import datetime 
current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

results = f"/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_{current_datetime}"
if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")


ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces/run_HGF_rxinfer.ssub'

stdout_name = f"{results}/logs/EF-%J.stdout"
stderr_name = f"{results}/logs/EF-%J.stderr"

predictions_or_responses = "responses"

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/emotional_faces_prolific_IDs.csv'
subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        subjects.append(line.strip())

for subject in subjects[2:3]:
    jobname = f'EF-RxInfer-{subject}'

    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results} {subject} {predictions_or_responses}")

    print(f"SUBMITTED JOB [{jobname}]")

#python3 /media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces/HGF_loop.py