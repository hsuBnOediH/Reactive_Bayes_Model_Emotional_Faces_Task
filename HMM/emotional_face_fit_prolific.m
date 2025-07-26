function [fit_results, DCM] = emotional_face_fit_prolific(subject,folder,params,field, plot, model)
    directory = dir(folder);
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    [~, sortedIndices] = sort(dates);
    sortedDirectory = directory(sortedIndices);
    
    % mask1: matches the subject base name
    base_mask = contains({sortedDirectory.name}, [subject]);
    % combine masks
    index_array = find(base_mask);

    if isempty(index_array)
        error('No matching files found for subject %s in %s mode.', subject, params.mode);
    elseif numel(index_array) > 1
        disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
        index_array = index_array(1);
    end

    file_index = index_array(1);
    file = '';
    file = [folder '/' sortedDirectory(file_index).name];

    subdat = readtable(file);
    switch params.mode
        case 'response'
            timeVar   = 'response_response_time';
            choiceVar = 'response_response';
            resultVar = 'response_result';
        case 'prediction'
            timeVar   = 'predict_response_time';
            choiceVar = 'predict_response';
            resultVar = 'predict_result';
        otherwise
            error('Unknown mode "%s" – must be ''response'' or ''prediction''.', params.mode);
    end
    % 1 based indexing, 200 trials in total
    trial_num = subdat.trial_number;
    % face stimulus type, could be sad or angry, converted to numeric, angry is 0 and sad is 1
    face_type = string(subdat.stim_type);
    face_type = double( face_type == "sad" );
    % observed tone type, could be high or low, converted to numeric, high is 1 and low is 0
    tone_type = string(subdat.tone);
    tone_type = double( tone_type == "high" );
    % association type, subject's belief about the association between face and tone
    % could either be sad_high-angry_low or sad_low-angry_high, 
    % in numeric, sad_high-angry_low is 1 and sad_low-angry_high is 0
    association = double( face_type == tone_type );

    % observed tone intensity, high or low, converted to numeric, high is 1 and low is 0.5
    intensity = string(subdat.intensity);
    intensity = double( intensity == "high" ) + 0.5 * double( intensity == "low" );
    % unclear what this is, but it is a numeric value 1 or 0
    expectation = subdat.expectation;
    % ground truth of the association, could be sad_high-angry_low
    prob_hightone_sad = subdat.prob_hightone_sad;
    % the gender of face stimulus, could F or M, converted to numeric, M is 1 and F is 0
    gender = string(subdat.gender);
    gender = double(gender == "M");
    % the block index, 1 to 6, each block size is dynamic, could from 10 to 50
    block_index = subdat.block_index;
    % the maximun seconds for answering the question, exceeding this time will be considered as no response
    response_duration = subdat.response_duration;
    % the actual response time, in seconds, if for the trail the subject did not respond, it will be NaN
    response_time = subdat.(timeVar);
    % the actual response, angry or sad indicated the face type, converted to numeric, angry is 0 and sad is 1
    % no response will be NaN
    resp_str = string(subdat.(choiceVar));  
    response = nan(size(resp_str));           
    response( resp_str=="angry" ) = 0;        
    response( resp_str=="sad"   ) = 1;    
    % the actual result of the response, 1 for correct and 0 for incorrect, NaN for no response
    result = subdat.(resultVar);
    % the actual reward, correct response will get reward between 75 and -25, depending on the rt
    % incorrect response will get -50, and no response will get -200
    reward = subdat.reward;

    trial_info = struct( ...
    'trial_num',    trial_num, ...
    'face_type',    face_type, ...
    'tone_type',    tone_type, ...
    'association',  association, ...
    'intensity',    intensity, ...
    'expectation',  expectation, ...
    'prob_hightone_sad', prob_hightone_sad, ...
    'gender',       gender, ...  
    'block_index',  block_index ...
    );

    % observation should include face_type, tone_type, intensity, result, response, reward
    % action should include response

    for n = 1:size(trial_num, 1)
        o{n} = [face_type(n) tone_type(n) intensity(n) gender(n) response(n) response_time(n) reward(n)];
        u{n} = [response(n)];
    end

    DCM.field  = field;            % Parameter field
    DCM.trial_info = trial_info; % Trial information
    DCM.U      =  o;                % observed outcomes (stimulus)
    DCM.Y      =  u;                % responses (actions)
    % DCM.reaction_times = reaction_times;
    DCM.params = params;
    DCM.mode = 'fit';
    DCM        = emotional_face_inversion(DCM, model);   % Invert the model

    fields = fieldnames(DCM.M.pE);

    for i = 1:length(fields)
        field = fields{i};
        if ismember(field, bounded01_fields)
            params.(field) = 1/(1+exp(-DCM.Ep.(field)));
        elseif ismember(field, logspace_fields)
            params.(field) = exp(DCM.Ep.(field));
        else
            params.(field) = DCM.Ep.(field);
        end
    end

    all_MDPs = [];
    act_prob = []
    model_acc = [];

    u = DCM.U;
    y = DCM.Y;

    num_trials = size(u,2);


    %priors = posteriors;
    %MDP     =
    %advise_gen_model(trialinfo(30*idx_block-29:30*idx_block,:),priors);
    %old model
    if model == 4
        MDP     = advise_gen_model_uni(trialinfo(30*idx_block-29:30*idx_block,:),params);
    end
       
    if (num_trials == 1)
        outcomes = u;
        actions = y;
        MDP.o  = outcomes{1};
        MDP.u  = actions{1};
        MDP.actualreward  = actualrewards(1);
    else
        outcomes = u(30*idx_block-29:30*idx_block);
        actions  = y(30*idx_block-29:30*idx_block);
        actualreward = actualrewards(30*idx_block-29:30*idx_block);

        for idx_trial = 1:30
            MDP(idx_trial).o = outcomes{idx_trial};
            MDP(idx_trial).u = actions{idx_trial};
            MDP(idx_trial).actualreward = actualreward(idx_trial);
            MDP(idx_trial).reaction_times = DCM.reaction_times{idx_trial};
            task.true_p_right(idx_trial) = 1-str2double(trialinfo{(idx_block-1)*30+idx_trial,2});
            task.true_p_a(idx_trial) = str2double(trialinfo{(idx_block-1)*30+idx_trial,1});
        end

        if strcmp(trialinfo{idx_block*30-29,3}, '80')
            task.block_type = "LL";
        else
            task.block_type = "SL";
        end
    end

    % solve MDP and accumulate log-likelihood
    %--------------------------------------------------------------------------
    if model == 1
        MDPs  = active_inference_model_uni(task, MDP, params, 0);
    elseif model == 2
        MDPs  = rl_model_connect_uni(task, MDP,params, 0);
    elseif model == 3
        MDPs  = rl_model_disconnect_uni(task, MDP,params, 0);
    elseif model == 4
        MDPs  = active_inference_model_mp_uni(task, MDP, params, 0);
    else
        error("model not recognized")
    end

    for j = 1:numel(actions)
        % if not choice the hint
        if actions{j}(2,1) ~= 2 
            if model == 4
                gt_aciton_idx = actions{j}(2,1)-1;
                time1_action_prob = MDPs(j).P(1,2:4);
                action_prob = time1_action_prob(gt_aciton_idx);
                act_prob_time1 = [act_prob_time1 action_prob];
                if  action_prob == max(time1_action_prob)
                    model_acc_time1 = [model_acc_time1 1];
                else
                    model_acc_time1 = [model_acc_time1 0];
                end
            else
                action_prob = MDPs.blockwise.action_probs(actions{j}(2,1)-1,1,j);
                act_prob_time1 = [act_prob_time1 action_prob]; 
                if action_prob == max(MDPs.blockwise.action_probs(:,1,j))
                    model_acc_time1 = [model_acc_time1 1];
                else
                    model_acc_time1 = [model_acc_time1 0];
                end
            end
        else
            if model ==4
                gt_time1_aciton_idx = actions{j}(2,1)-1;
                time1_action_prob = MDPs(j).P(1,2:4);
                gt_time2_aciton_idx = actions{j}(2,2)-2;
                prob_choose_advisor = MDPs(j).P(1, 2, 1);

                time2_action_prob = MDPs(j).P(1,7:8);
                prob_choose_bandit = time2_action_prob(gt_time2_aciton_idx);

                act_prob_time1 = [act_prob_time1 prob_choose_advisor];
                act_prob_time2 = [act_prob_time2 prob_choose_bandit];

                if prob_choose_advisor == max(time1_action_prob)
                    model_acc_time1 = [model_acc_time1 1];
                else
                    model_acc_time1 = [model_acc_time1 0];
                end

                if prob_choose_bandit==max(time2_action_prob)
                    model_acc_time2 = [model_acc_time2 1];
                else
                    model_acc_time2 = [model_acc_time2 0];
                end  
            else

                prob_choose_advisor = MDPs.blockwise.action_probs(1,1,j); 
                prob_choose_bandit = MDPs.blockwise.action_probs(actions{j}(2,2)-1,2,j);
                act_prob_time1 = [act_prob_time1 prob_choose_advisor];
                act_prob_time2 = [act_prob_time2 prob_choose_bandit];
                if prob_choose_advisor==max(MDPs.blockwise.action_probs(:,1,j))
                    model_acc_time1 = [model_acc_time1 1];
                else
                    model_acc_time1 = [model_acc_time1 0];
                end

                if prob_choose_bandit==max(MDPs.blockwise.action_probs(:,2,j))
                    model_acc_time2 = [model_acc_time2 1];
                else
                    model_acc_time2 = [model_acc_time2 0];
                end   
            end
        end
    end

    all_MDPs = [all_MDPs; MDPs'];
    clear MDPs


    if plot && model~=4
        for i=1:length(DCM.U)
            MDP(i).o = DCM.U{1,i};
            MDP(i).u = DCM.Y{1,i};
            MDP(i).reaction_times = DCM.reaction_times{1,i};
            
            block_num = ceil(i/30);
            trial_num_within_block = i - (block_num-1)*30;
            trial_action_probs = all_MDPs(block_num).blockwise.action_probs(:,:,trial_num_within_block);
            % Concatenate the zero row at the top of the matrix
            zero_row = zeros(1, size(trial_action_probs, 2));
            trial_action_probs = vertcat(zero_row, trial_action_probs)';
            MDP(i).P = permute(trial_action_probs, [3 2 1]);
        end
        advise_plot_uni(MDP);
    end
        
    fit_results.id = subject;
    fit_results.file = file;
    % assign priors/posteriors/fixed params to fit_results
    param_names = fieldnames(params);
    for i = 1:length(param_names)
        % param was fitted
        if ismember(param_names{i}, fields)
            fit_results.(['posterior_' param_names{i}]) = params.(param_names{i});
            fit_results.(['prior_' param_names{i}]) = DCM.params.(param_names{i});  
        % param was fixed
        else
            fit_results.(['fixed_' param_names{i}]) = params.(param_names{i});

        end
    end

    fit_results.avg_act_prob_time1 = sum(act_prob_time1)/length(act_prob_time1);
    fit_results.avg_act_prob_time2 = sum(act_prob_time2)/length(act_prob_time2);
    fit_results.avg_model_acc_time1   = sum(model_acc_time1)/length(model_acc_time1);
    fit_results.avg_model_acc_time2   = sum(model_acc_time2)/length(model_acc_time2);
    fit_results.times_chosen_advisor = length(model_acc_time2);
end


