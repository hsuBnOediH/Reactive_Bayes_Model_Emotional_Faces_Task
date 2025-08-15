
function [DCM] = emotional_face_inversion(DCM, model)
    ALL = false;
    prior_variance = .5;
    %define the global variables for all the functions in this file
    global bounded01_fields logspace_fields;
    
    bounded01_fields = DCM.params.fields_normal_range.bounded01_fields;
    logspace_fields = DCM.params.fields_normal_range.logspace_fields;
   


    for i = 1:length(DCM.field)
        field = DCM.field{i};
        if ALL
            pE.(field) = zeros(size(param));
            pC{i,i}    = diag(param);
        else
            if ismember(field, bounded01_fields)
                % bound-[0,1] transform
                pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));
                pC{i,i}    = prior_variance;
            elseif ismember(field, logspace_fields)
                % log-positive transform
                pE.(field) = log(DCM.params.(field));
                pC{i,i}    = prior_variance;
            else
                % leave untouched
                pE.(field) = DCM.params.(field);
                pC{i,i}    = prior_variance;
            end
        end
    end

    pC = spm_cat(pC);

    % model specification
    %--------------------------------------------------------------------------
    M.model = model;
    M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
    M.pE    = pE;                            % prior means (parameters)
    M.pC    = pC;                            % prior variance (parameters)
    M.mode  = DCM.mode;
    M.params = DCM.params;
    M.trial_info = DCM.trial_info;           % trial information
    % Variational Laplace
    %--------------------------------------------------------------------------
    [Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

    % Store posterior densities and log evidence (free energy)
    %--------------------------------------------------------------------------
    DCM.M   = M;
    DCM.Ep  = Ep;
    DCM.Cp  = Cp;
    DCM.F   = F;
return

function L = spm_mdp_L(P,M,U,Y)
    global bounded01_fields logspace_fields;

    % log-likelihood function
    % FORMAT L = spm_mdp_L(P,M,U,Y)
    % P    - parameter structure
    % M    - generative model
    % U    - inputs
    % Y    - observed repsonses
    %__________________________________________________________________________

    if ~isstruct(P); P = spm_unvec(P,M.pE); end

    % multiply parameters in MDP
    %--------------------------------------------------------------------------
    %mdp   = M.mdp;
    fields = fieldnames(M.pE);
    params = M.params;
    for i = 1:numel(fields)
        field = fields{i};

        if ismember(field, bounded01_fields)
            % sigmoid‐inverse transform
            params.(field) = 1./(1 + exp(-P.(field)));

        elseif ismember(field, logspace_fields)
            % positive log‐space transform
            params.(field) = exp(P.(field));

        else
            % no transform
            params.(field) = P.(field);
        end
    end
    L = 0;
    
    if M.model == 1
        MDP     = emotional_face_gen_model(M.trial_info,params);
    end
    
    % MDP structure
    %--------------------------------------------------------------------------
    % observation should include: face_type, tone_type, intensity, result, response, reward
    for idx_trial = 1:200
        MDP(idx_trial).o = U(idx_trial);
        MDP(idx_trial).u = Y;
    end
    
    task.field = fields;

    if M.model == 1
        MDP  = active_inference_model_mp(task, MDP,params, 0);
    else
        print('Model not recognized')
    end


    for j = 1:200
        if actions{j}(2,1) ~= 2
            % TODO: still use the old model? to compute the log likelihood?
            if M.model == 4
                L = L + log(MDP(j).P(1,actions{j}(2,1),1) + eps); %old model
            else
                %L = L + log(MDP(j).P(1,actions{j}(2,1),1) + eps); old model
                prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,1)-1,1,j); 
                L = L + log(prob_choose_bandit + eps);
            end
        else % when advisor was chosen
            if M.model == 4
                prob_choose_advisor = MDP(j).P(1,actions{j}(2,1),1);
                L = L + log(prob_choose_advisor + eps);
                prob_choose_bandit = MDP(j).P(1,actions{j}(2,2),2);
                L = L + log(prob_choose_bandit + eps);
            else
                prob_choose_advisor = MDP.blockwise.action_probs(1,1,j); 
                %prob_choose_advisor = MDP(j).P(1,actions{j}(2,1),1); old model
                L = L + log(prob_choose_advisor + eps);
                prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,2)-1,2,j); 
                %prob_choose_bandit = MDP(j).P(1,actions{j}(2,2),2); old model
                L = L + log(prob_choose_bandit + eps);
            end
        end
    end

    clear('MDP')


  

    fprintf('LL: %f \n',L)
return


