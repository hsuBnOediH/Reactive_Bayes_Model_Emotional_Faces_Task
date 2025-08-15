% Authors: Feng, July 2024
% 
% Main Script for HMM model for the Emotional Face Task

%{
Task overview:
two-hundred-trial “emotional face” experiment in which a participant hears a low or high tone, 
then sees a face whose expression varies from clearly sad to clearly angry. 
They must decide “sad” or “angry” as quickly as possible; correct, fast responses yield higher reward, 
while incorrect or too-slow responses incur minimal reward. The tone–face association changes unpredictably during the session, 
so the model must continuously relearn which tone maps to which emotion.


o(observations): participants' responses to the emotional faces is sad or angry.
s(hiden states): the hidden states of which tone-face mapping is currently active.




A(likelihood): P(observation | hidden state)-> P(o)
    How likely each observation (o) is under each hidden state (s).
B(transition): P(hidden state | previous hidden state) -> P(s|s_prev)
    How likely the hidden state is to transition from one to another.
C(preferences): Log-preference over observation, however in this case, we do not have preferences, so it is empty.
D(priors over initial hidden state): P(hidden state) -> P(s)
    The prior over the initial hidden state. In other words, in the first trial, which association is more likely to be active in subjects' mind.
%}



dbstop if error
rng('default');
clear all;
clear variables;
plot = true;

ON_CLUSTER = getenv('ON_CLUSTER');

env_sys = '';
if ispc
    env_sys = 'pc';
elseif ismac
    env_sys = 'mac';
elseif isunix
    env_sys = 'cluster';
else
    disp('Unknown operating system.');
end
% SWITCHES
% True -> Generate simulated behavior
SIM = false;
if ON_CLUSTER
    SIM = getenv('SIM');
    % tanform the char to logical
    SIM = strcmp(SIM, 'True');
end
% True -> Fit the behavior data into the model
FIT = true;
if ON_CLUSTER
    FIT = getenv('FIT');
    % tanform the char to logical
    FIT = strcmp(FIT, 'True');
end 

PLOT = true;
if ON_CLUSTER
    PLOT = getenv('PLOT');
end 

% SETTINGS
% Subject identifier for the test or experiment, if on cluster read from ENV var
FIT_SUBJECT = '5db4ef4a2986a3000be1f886';
if ON_CLUSTER
    FIT_SUBJECT = getenv('FIT_SUBJECT');
end

% ROOT:
% If ROOT is not assigned (i.e., empty), the script will derive the root 
% path based on the location of the main file.
ROOT = ''; 
if isempty(ROOT)
    ROOT = fileparts(mfilename('fullpath'));
    disp(['ROOT path set to: ', ROOT]);
end


% RES_PATH:
% If RES_PATH is not assigned (i.e., empty), it will be auto-generated relative to ROOT.
% If RES_PATH is a relative path, it will be appended to the ROOT path.
RES_PATH = '../outputs/HMM/debug/';
if ON_CLUSTER
    RES_PATH = getenv('RES_PATH');
end

% INPUT_PATH:
% The folder path where the subject file is located. If INPUT_PATH is a relative path,
% it will be appended to the ROOT path.
INPUT_PATH = '../outputs/processed_data/debug/';
if ON_CLUSTER
    INPUT_PATH = getenv('INPUT_PATH');
end

% MODEL_IDX:
% only have model 1 for HMM for emotional face task
MODEL_IDX = 1; 
if ON_CLUSTER
    env_value = getenv('MODEL_IDX');
    MODEL_IDX= str2double(env_value);
end




% Display all settings and switches
disp('--- Settings and Switches ---');
disp(['SIM (Simulate Behavior): ', num2str(SIM)]);
disp(['FIT (Fit Behavior Data): ', num2str(FIT)]);
disp(['PLOT (Plot Results): ', num2str(PLOT)]);
disp(['ON_CLUSTER (Example Subject): ', ON_CLUSTER]);
disp(['FIT_SUBJECT (Subject Identifier): ', FIT_SUBJECT]);
disp(['ROOT Path: ', ROOT]);
disp(['RES_PATH (Results Path): ', RES_PATH]);
disp(['INPUT_PATH (Input Path): ', INPUT_PATH]);
disp(['Environment System: ', env_sys]);
disp(['MODEL_IDX: ', num2str(MODEL_IDX)]);
disp('-----------------------------');

% Add external paths depending on the system
if strcmp(env_sys, 'pc')
    spmPath = 'L:/rsmith/all-studies/util/spm12/';
    spmDemPath = 'L:/rsmith/all-studies/util/spm12/toolbox/DEM/';
    tutorialPath = 'L:/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
   
elseif strcmp(env_sys, 'mac')
    spmPath =  [ROOT '/../../spm/'];
    spmDemPath = [ROOT '/../../spm/toolbox/DEM/'];
    tutorialPath = [ROOT '/../../Active-Inference-Model-for-Advise-Task/Active-Inference-Tutorial-Scripts-main'];

    INPUT_DIRECTORY = [ROOT '/' INPUT_PATH];

elseif strcmp(env_sys, 'cluster')
    spmPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12';
    spmDemPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12/toolbox/DEM';
    tutorialPath = '/mnt/dell_storage/labs/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
    INPUT_DIRECTORY = [ROOT '/' INPUT_PATH];
end

addpath(spmPath);
addpath(spmDemPath);
addpath(tutorialPath);

bounded01_fields = { ...
    'p_hs_la','p_correct','p_stay'...
};

logspace_fields = { ...
    'inv_temp'...
};


if SIM
   % implement the simulation part later

end

if FIT
    params.fields_normal_range.bounded01_fields = bounded01_fields;
    params.fields_normal_range.logspace_fields = logspace_fields;
    params.mode = 'response'; %could be 'response' or 'prediction'
    params.p_hs_la = 0.5; % probability of high tone given sad face and low tone given angry face
    params.p_high_intensity = 0.7; % probability of correct response given the correct association
    params.p_low_intensity = 0.3; % probability of staying in the same state
    % implement the field part later
    field = {'p_hs_la','p_high_intensity','p_low_intensity'};
    [fit_results, DCM] = emotional_face_fit_prolific(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot, MODEL_IDX);
    model_free_results = emotional_face_mf_uni(fit_results.file);
    mf_fields = fieldnames(model_free_results);
    for i=1:length(mf_fields)
        fit_results.(mf_fields{i}) = model_free_results.(mf_fields{i});      
    end

    fit_results.F = DCM.F;
    fit_results.modelAIorRL = MODEL_IDX;
    if ~ONEMODEL
        results_dir = fullfile(ROOT, RES_PATH);
        % Define the folder name dynamically based on paramcombi
        folder_name = fullfile(results_dir, ['paramcombi' num2str(paramcombi)]);
        % Check if the folder exists; if not, create it
        if ~exist(folder_name, 'dir')
            mkdir(folder_name);
        end
        % Save the table to the folder
        writetable(struct2table(fit_results), ...
            fullfile(folder_name, ['emo_face-' FIT_SUBJECT  '_fits.csv']));
        % Save the plot to the folder
        saveas(gcf, fullfile(folder_name, [FIT_SUBJECT  '_fit_plot.png']));
        % Save the .mat file to the folder
        save(fullfile(folder_name, ['fit_results_' FIT_SUBJECT  '.mat']), 'DCM');
    else
        writetable(struct2table(fit_results), [results_dir '/emo_face-' FIT_SUBJECT '_fits.csv']); 
        saveas(gcf,[results_dir '/' FIT_SUBJECT '_fit_plot.png']);
        save(fullfile([results_dir '/fit_results_' FIT_SUBJECT '.mat']), 'DCM');
    end
end

    
    
    
    
    
    