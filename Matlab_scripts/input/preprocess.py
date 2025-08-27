# read the schedule file(both CB1 and CB2)
import pandas as pd
import os
import datetime
from tqdm import tqdm
schedule_file_cb1_path = 'schedule/emotional_faces_CB1_schedule.csv'
schedule_file_cb2_path = 'schedule/emotional_faces_CB2_schedule.csv'
schedule_file_cb1 = pd.read_csv(schedule_file_cb1_path)
schedule_file_cb2 = pd.read_csv(schedule_file_cb2_path)

# list all the subjects data list
subject_data_folder_path = 'data/'
subject_data_files_list = os.listdir(subject_data_folder_path)

dt = datetime.datetime.now()
output_folder_path = f"../outputs/processed_data/{dt.strftime('%Y-%m-%d_%Hh%M.%S')}"
# create the output folder if not exists
os.makedirs(output_folder_path, exist_ok=True)


for file in tqdm(subject_data_files_list, desc="Processing subject data files"):
    if file.endswith('.csv'):
        file_name_split = file.split('_')
        # example emotional_faces_v2_5db4ef4a2986a3000be1f886_T2_CB_2024-06-22_11h26.23.192.csv
        subject_id = file_name_split[3]
        is_cb2 = file_name_split[5] == 'CB'
        if is_cb2:
            schedule_file = schedule_file_cb2
        else:
            schedule_file = schedule_file_cb1
        # read the subject data file as dataframe
        subject_data_file_path = os.path.join(subject_data_folder_path, file)
        subject_data = pd.read_csv(subject_data_file_path)
        # remove all the rows before the last row, which trial_type is  'MAIN'
        # drop everything before the last “MAIN” trial
        if 'trial_type' in subject_data.columns:
            main_mask = subject_data['trial_type'] == 'MAIN'
            if main_mask.any():
                # find the index of the last MAIN
                last_main_idx = main_mask[main_mask].index[-1]
                # keep from that row onward
                subject_data = subject_data.iloc[last_main_idx+1:].reset_index(drop=True)
            else:
                # no MAIN rows found — decide how to handle (e.g. keep all or skip this subject)
                pass
        else:
            raise KeyError("`trial_type` column not found in subject_data")
        # output the processed data to path
        output_file_name = f'{subject_id}_processed.csv'
        output_file_path = os.path.join(output_folder_path, output_file_name)

        # init the output dataframe with columns
        # trial_number, stim_type, tone, intensity, expectation, prob_hightone_sad,gender,block_index,response_duration,
        # response_time,response,result, reward

        rows = []
        for trial in range(1, 201):  # trials 1–200
            # schedule info
            schedule_row = schedule_file[schedule_file['trial_number'] == trial].iloc[0]
            # subject info
            # in subject_data, filter the rows with trial_number == trial-1, the index is 0-based!!
            subject_rows = subject_data[subject_data['trial'] == trial-1]

            predict_row = subject_rows[subject_rows['event_type'] == 12]
            responses_row = subject_rows[subject_rows['event_type'] == 7]
            feedback_row = subject_rows[subject_rows['event_type'] == 9]

            # predict info
            predict_rt = predict_row['response_time'].iloc[0] if not predict_row.empty else None
            #  left (angry) or right (sad) key pressed
            predict_response = predict_row['response'].iloc[0] if not predict_row.empty else None
            predict_response = 'angry' if predict_response == 'left' else 'sad' if predict_response == 'right' else None
            predict_result = predict_row['result'].iloc[0].lower() if not predict_row.empty else None
            predict_result = 1 if predict_result == 'correct' else 0 if predict_result == 'incorrect' else None

            # response info
            response_rt = responses_row['response_time'].iloc[0] if not responses_row.empty else None
            response = responses_row['response'].iloc[0] if not responses_row.empty else None
            response = 'angry' if response == 'left' else 'sad' if response == 'right' else None
            response_result = responses_row['result'].iloc[0].lower() if not responses_row.empty else None
            response_result = 1 if response_result == 'correct' else 0 if response_result == 'incorrect' else None
            reward = feedback_row['response'].iloc[0] if not feedback_row.empty else None
            reward = int(reward) if reward is not None else None

            rows.append({
                'trial_number': trial,
                'stim_type': schedule_row['stim_type'],  # e.g. 'sad' / 'angry'
                'tone': schedule_row['tone'],  # e.g. 'high' / 'low'
                'intensity': schedule_row['intensity'],  # e.g. 'high' / 'low'
                'expectation': schedule_row['expectation'],  # 0 or 1
                'prob_hightone_sad': schedule_row['prob_hightone_sad'],  # ground truth
                'gender': schedule_row['gender'],  # 'F' or 'M'
                'block_index': schedule_row['block_num'],  # 1–6
                'response_duration': schedule_row.get('response_duration', None),
                'predict_response_time': predict_rt,
                'predict_response': predict_response,
                'predict_result': predict_result,
                'response_response_time': response_rt,
                'response_response': response,
                'response_result': response_result,
                'reward': reward,
            })

            # turn into DataFrame and write out
        res_df = pd.DataFrame(rows)
        output_file_name = f'{subject_id}_processed_{"CB2" if is_cb2 else "CB1"}.csv'
        res_df.to_csv(os.path.join(output_folder_path, output_file_name), index=False)

    print("Done! Processed files are in:", output_folder_path)




