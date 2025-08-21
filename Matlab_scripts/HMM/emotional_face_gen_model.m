function mdp = emotional_face_gen_model(trialinfo, priors)
    % Unpack priors
    p_hs_la = priors.p_hs_la;  % Probability of high tone given sad face (context reliability)
    prior_d  = 0.5;                % Prior Dirichlet count for initial context beliefs (can adjust if needed)
    p_high_intensity = priors.p_high_intensity; % Chance the face looks sad if it is truly sad, under high intensity case
    p_low_intensity = priors.p_low_intensity; % Chance the face looks sad if it is truly sad, under low intensity case
    times_counts = 200;  % Concentration counts for A matrix (if learning A, higher = more confidence in prior)
    
    % Number of trials
    nTrials = size(trialinfo.trial_num, 1);
    mdp(nTrials) = struct();  % preallocate struct array
    
    for t = 1:nTrials
        % Extract trial-specific info
        face_type       = trialinfo.face_type(t);        % 1 = sad face, 0 = angry face (actual face shown)
        prob_hightone_sad = trialinfo.prob_hightone_sad(t); % True probability of high tone given sad face (context reliability)
        intensity       = trialinfo.intensity(t);        % Tone intensity (1 or 0.5)
        tone_type       = trialinfo.tone_type(t);        % 1 = high tone presented, 0 = low tone presented
        



        % if facetype is angry, then context1 of D{1} is 1, otherwise context2 is 1
        % Prior belief distribution over context states [context1; context2]
        % TODO: Time count here or not???
        D{1} = [prior_d, 1-prior_d];  % Dirichlet prior for context (2 states)

        %% Define outcome modalities and hidden state dimensions
        % Only hidden state factor: Context (2 states)
        Ns(1) = 2;                 
        
        % Outcome modality 1: Tone Modality (low/high tone and null states)
        % Outcome modality 2: Face Modality (sad/angry face and null states)
        No(1) = 3;
        No(2) = 3;
        
        %% Likelihood (A matrices)
        % Initialize A matrices with correct dimensions
        A{1} = zeros(No(1), Ns(1));  % 3 x 2
        
        % in A{1}, we have:
        % - Row 1: Null States(NaN)
        % - Row 2: Low tone (0)
        % - Row 3: High tone (1)
        % - Column 1: Context state 1 (angry face)
        % - Column 2: Context state 2 (sad face)
        % according to the face_type and tone_type, fill in the corresponding probabilities to 1, leaving the rest as 0.
        A{1}(1, 1) = 0.5;
        A{1}(1, 2) = 0.5;
        A{1}(2, 1) = p_hs_la;  % Given Low tone, the face is angry
        A{1}(2, 2) = 1 - p_hs_la;  % Given Low tone, the face is sad
        A{1}(3, 1) = 1 - p_hs_la;  % Given High tone, the face is angry
        A{1}(3, 2) = p_hs_la;  % Given High tone, the face is sad
      

        % prepare two sets of A{2} for Face outcome modality
        % One for High Instensity, one for Low Intensity, both follow the same structure:
        % - Row 1: Null States(NaN)
        % - Row 2: Angry face (0)
        % - Row 3: Sad face (1)
        % - Column 1: Context state 1 (Angry face)
        % - Column 2: Context state 2 (Sad face)
        A{2} = zeros(No(2), Ns(1));  % 3 x 2

        low_intensity_a_2 = zeros(No(2), Ns(1));  % 3 x 2 for low intensity
        high_intensity_a_2 = zeros(No(2), Ns(1));  % 3 x 2 for high intensity
        
        if intensity == 0.5  % Low intensity
            low_intensity_a_2(1, 1) = 0.5;
            low_intensity_a_2(1, 2) = 0.5;
            low_intensity_a_2(2, 1) = p_low_intensity;  % Angry face in context1 (Angry)
            low_intensity_a_2(2, 2) = 1 - p_low_intensity;  % Sad face in context1 (Sad)
            low_intensity_a_2(3, 1) = 1 - p_low_intensity;  % Angry face in context2 (Angry)
            low_intensity_a_2(3, 2) = p_low_intensity;  % Sad face in context2 (Sad)
            A{2} = low_intensity_a_2;  % Use low intensity A{2}
        else  % High intensity
            high_intensity_a_2(1, 1) = 0.5;
            high_intensity_a_2(1, 2) = 0.5;
            high_intensity_a_2(2, 1) = p_high_intensity;  % Angry face in context1 (Angry)
            high_intensity_a_2(2, 2) = 1 - p_high_intensity;  % Sad face in context1 (Sad)
            high_intensity_a_2(3, 2) = p_high_intensity;  % Sad face in context2 (Sad)
            high_intensity_a_2(3, 1) = 1 - p_high_intensity;  % Angry face in context2 (Angry)
            A{2} = high_intensity_a_2;  % Use high intensity A{2}
        end
        
        %% Transition probabilities (B matrices)
        % B{1}: Context transitions (2x2)
        % All hidden state transitions can be set to identity (no change), since all the state no change within Trail
        B{1} = [1, 0; 0, 1];


        %% Preferences (C matrices)
        % No explicitly optimizing reward within the model
        % C{1} is for tone modality, C{2} is for face modality
        T = 3;  % number of time steps within a trial (observation and then feedback)
        C{1} = zeros(No(1), T);
        C{2} = zeros(No(2), T);
    
        %% Policies (V)
        % no alternate actions or control states
        % only one “policy,” which is essentially to passively observe the tone then the face.
        Np = 2;  % number of policies
        V = [1,1];  % both policies are active


        %% Observations (O matrices)
        % If you have Ng modalities and T time‐points, each
        % MDP(m).O{g} is an No(g)×T matrix, where No(g) is
        % the number of possible outcomes in modality g.
        Ng = 2;  % number of modalities (tone and face)
        O = cell(1, Ng);
        for g = 1:Ng
            O{g} = zeros(No(g), T);
        end
        % At time step 1, nothing is observed yet, so as null state
        O{1}(1, 1) = 1;  
        O{2}(1, 1) = 1;  
        % At time step 2, the tone is observed, init according to the
        % tone_type
        if tone_type == 0
            % if low tone observed 
            O{1}(2, 2) = 1;
        else
            % if high tone observed 
            O{1}(3, 2) = 1;
        end
        % face type is still nan state at time 2
        O{2}(1, 2) = 1;

        % At time step 3, the the tone is observed, init according to the
        % tone_type
        % face type is observed also, sampling posteriors over states
        if tone_type == 0
            % if low tone observed 
            O{1}(2, 3) = 1;
        else
            % if high tone observed 
            O{1}(3, 3) = 1;
        end

        % Sample the observed face type from prob
        
        % compute the prob of face
        p_hidden = D{1};          % 1×2 vector of hidden-state probabilities
        likelihood_matrix = A{2};          % 3×2 likelihood matrix

        % 1) sample hidden state index h ∈ {2,3}
        r = rand();
        hidden_state_index = find(r <= cumsum(p_hidden), 1);
        ob_face_given_hidden_state = likelihood_matrix(2:3, hidden_state_index);

        % 2) sample observed face index o ∈ {1,2} given hidden = h
        r = rand();
        ob_face_type_index = find(r <= cumsum(ob_face_given_hidden_state), 1);
        
        
        if ob_face_type_index == 1
            % angry face
            O{2}(2, 3) = 1;
        else
            % sad face
            O{2}(3, 3) = 1;
        end

        %% Populate mdp structure for this trial
        mdp(t).A = A;
        mdp(t).B = B;
        mdp(t).C = C;
        mdp(t).D = D;
        mdp(t).T = T;
        mdp(t).U = V;


        mdp(t).O = O;

        % Save some parameters for reference (not used by spm_MDP_VB, but can be useful for analysis)
        mdp(t).prior_d   = prior_d;
        mdp(t).p_hs_la   = p_hs_la;
        mdp(t).p_high_intensity = p_high_intensity;
        mdp(t).p_low_intensity = p_low_intensity;
    end
    
    % Check model consistency for all trials
    mdp = spm_MDP_check(mdp);
end