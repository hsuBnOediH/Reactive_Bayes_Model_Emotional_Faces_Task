function [fit_results, DCM] = emotional_face_fit_prolific(subject,folder,params,field, plot, model)
    directory = dir(folder);
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    [~, sortedIndices] = sort(dates);
    sortedDirectory = directory(sortedIndices);
    
    % mask1: matches the subject base name
    base_mask = contains({sortedDirectory.name}, ['task_data_' subject]);

    % mask2: matches the desired suffix based on params.mode
    switch params.mode
        case "response"
            suffix_mask = endsWith({sortedDirectory.name}, '_responses.csv');
        case "predictions"
            suffix_mask = endsWith({sortedDirectory.name}, '_predictions.csv');
        otherwise
            % if you ever want all .csv
            suffix_mask = endsWith({sortedDirectory.name}, '.csv');
    end

    % combine masks
    index_array = find(base_mask & suffix_mask);

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
    data.trial     = subdat.trial_number;
    data.response  = subdat.response;
    data.observed  = subdat.observed;
    data.intensity = subdat.intensity;

    % win lose result?
    % reward?
    % RT? 


    for n = 1:size(data.observed,1)
        DCM.field  = field;            % Parameter field
        DCM.U      =  data;              % trial specification (stimuli)
        DCM.Y      =  data;              % responses (action)
        % DCM.reaction_times = reaction_times;
        DCM.params = params;
        DCM.mode = 'fit';
        DCM        = emotional_face_inversion(DCM, model);   % Invert the model
        break;
    end

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


