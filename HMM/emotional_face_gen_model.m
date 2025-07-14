function mdp = emotional_face_gen_model(trialinfo, priors)
    % Unpack priors
    p_hs_la  = priors.p_hs_la;   % P(high tone | sad face) under Sad-High context (and low tone | angry face)
    p_correct = priors.p_correct; % Probability of correct feedback given the choice is correct
    p_stay   = priors.p_stay;    % Probability of staying in the same context on next trial
    prior_d  = 1;                % Prior Dirichlet count for initial context beliefs (can adjust if needed)

    times_counts = 200;  % Concentration counts for A matrix (if learning A, higher = more confidence in prior)
    
    % Number of trials
    nTrials = size(trialinfo, 1);
    mdp(nTrials) = struct();  % preallocate struct array
    
    for t = 1:nTrials
        % Extract trial-specific info
        face_type       = trialinfo.face_type(t);        % 1 = sad face, 0 = angry face (actual face shown)
        prob_hightone_sad = trialinfo.prob_hightone_sad(t); % True probability of high tone given sad face (context reliability)
        intensity       = trialinfo.intensity(t);        % Tone intensity (1 or 0.5)
        tone_type       = trialinfo.tone_type(t);        % 1 = high tone presented, 0 = low tone presented
        
        % Effective mapping probability for this trial, adjusted for intensity
        % (Pulls probability towards 0.5 if intensity is lower)
        p_eff = 0.5 + (p_hs_la - 0.5) * intensity;
        
        %% Prior belief over context (initial state D)
        % If we have a provided prior probability for context1 (prob_hightone_sad),
        % use that; otherwise default to 0.5.
        if ~isempty(prob_hightone_sad)
            % Ensure the value is between 0 and 1; if it's not provided or out of range, default to 0.5
            if isnan(prob_hightone_sad) || prob_hightone_sad < 0 || prob_hightone_sad > 1
                prob_context1 = 0.5;
            else
                % prob_hightone_sad might represent the probability that context is Sad-High (context1)
                prob_context1 = prob_hightone_sad;
            end
        else
            prob_context1 = 0.5;
        end
        % Prior belief distribution over context states [context1; context2]
        D{1} = [prob_context1; 1 - prob_context1];
        % Corresponding Dirichlet counts for learning (scaled by prior_d)
        d{1} = prior_d * D{1};
        
        %% Define outcome modalities and hidden state dimensions
        % Hidden state factor 1: Context (2 states)
        % Hidden state factor 2: Trial stage/choice (5 states as defined above)
        Nf = 2;                       % number of factors
        Ns(1) = 2;                    % states in context factor
        Ns(2) = 5;                    % states in trial stage factor
        
        % Outcome modality 1: Feedback (3 outcomes: null, lose, win)
        % Outcome modality 2: State observation (5 outcomes, one per trial state)
        No(1) = 3;
        No(2) = 5;
        
        %% Likelihood (A matrices)
        % Initialize A matrices with correct dimensions
        A{1} = zeros(No(1), Ns(1), Ns(2));  % 3 x 2 x 5
        A{2} = zeros(No(2), Ns(1), Ns(2));  % 5 x 2 x 5
        
        % Fill A{1}: Feedback outcomes
        % Case 1: Start state -> always Null outcome (no feedback yet)
        A{1}(1, :, 1) = 1;   % For both context states, if trial state = 1 (Start), outcome = Null with prob 1.
        A{1}(2, :, 1) = 0;
        A{1}(3, :, 1) = 0;
        
        % Cases 2-5: After a choice is made.
        % Loop over context states and trial states to set win/lose probabilities.
        % Context index 1 = Sad-High context, index 2 = Sad-Low context.
        for s1 = 1:2  % context state
            % Determine actual face emotion implied by context and tone:
            % If context1 (s1=1, Sad-High):
            %   high tone => face is Sad, low tone => face is Angry.
            % If context2 (s1=2, Sad-Low):
            %   high tone => face is Angry, low tone => face is Sad.
            % We use the actual tone presented (tone_type) from trialinfo.
            if s1 == 1  % Sad-High context
                true_face_if_high = 1;  % Sad
                true_face_if_low  = 0;  % Angry
            else        % Sad-Low context
                true_face_if_high = 0;  % Angry
                true_face_if_low  = 1;  % Sad
            end
            
            % Determine true face on this trial based on tone_type
            if tone_type == 1  % high tone presented
                true_face = true_face_if_high;
            else               % low tone presented
                true_face = true_face_if_low;
            end
            
            % Now assign probabilities for each choice state (2 to 5)
            % State 2: HighTone - Choose Sad
            % State 3: HighTone - Choose Angry
            % State 4: LowTone  - Choose Angry
            % State 5: LowTone  - Choose Sad
            % We check each state consistency with tone_type and choice.
            
            % State 2 (HighTone, Choose Sad) is only reachable if tone is high
            if tone_type == 1
                % Agent chose Sad, tone was high
                chosen_face = 1;  % agent's chosen label (Sad)
                % Check correctness
                if chosen_face == true_face
                    % Correct choice -> Win with p_correct, Lose with 1-p_correct
                    A{1}(3, s1, 2) = p_correct;       % Win outcome
                    A{1}(2, s1, 2) = 1 - p_correct;   % Lose outcome
                else
                    % Incorrect choice -> Lose with p_correct, (Win with 1-p_correct if any)
                    A{1}(2, s1, 2) = p_correct;
                    A{1}(3, s1, 2) = 1 - p_correct;
                end
            else
                % If tone was not high, this state wouldn't occur (set to null outcome deterministically)
                A{1}(1, s1, 2) = 1;  % we assign Null with probability 1 to indicate this state shouldn't actually happen
            end
            
            % State 3 (HighTone, Choose Angry)
            if tone_type == 1
                % Agent chose Angry, tone was high
                chosen_face = 0;  % agent's chosen label (Angry)
                if chosen_face == true_face
                    % Correct
                    A{1}(3, s1, 3) = p_correct;
                    A{1}(2, s1, 3) = 1 - p_correct;
                else
                    % Incorrect
                    A{1}(2, s1, 3) = p_correct;
                    A{1}(3, s1, 3) = 1 - p_correct;
                end
            else
                % Not reachable if tone was low
                A{1}(1, s1, 3) = 1;
            end
            
            % State 4 (LowTone, Choose Angry)
            if tone_type == 0
                % Agent chose Angry, tone was low
                chosen_face = 0;  % Angry
                if chosen_face == true_face
                    A{1}(3, s1, 4) = p_correct;
                    A{1}(2, s1, 4) = 1 - p_correct;
                else
                    A{1}(2, s1, 4) = p_correct;
                    A{1}(3, s1, 4) = 1 - p_correct;
                end
            else
                % Not reachable if tone was high
                A{1}(1, s1, 4) = 1;
            end
            
            % State 5 (LowTone, Choose Sad)
            if tone_type == 0
                % Agent chose Sad, tone was low
                chosen_face = 1;  % Sad
                if chosen_face == true_face
                    A{1}(3, s1, 5) = p_correct;
                    A{1}(2, s1, 5) = 1 - p_correct;
                else
                    A{1}(2, s1, 5) = p_correct;
                    A{1}(3, s1, 5) = 1 - p_correct;
                end
            else
                % Not reachable if tone was high
                A{1}(1, s1, 5) = 1;
            end
        end
        
        % After setting win/lose, ensure probabilities in each (s1, s2) slice sum to 1.
        % For unreachable states (where we set A{1}(Null)=1), they are already normalized.
        for s1 = 1:2
            for s2 = 2:5
                total = sum(A{1}(:, s1, s2));
                if total == 0
                    % If still zero (which could happen if not set above), set Null=1
                    A{1}(1, s1, s2) = 1;
                    total = 1;
                end
                A{1}(:, s1, s2) = A{1}(:, s1, s2) / total;
            end
        end
        
        % Apply tone mapping reliability (p_eff) to feedback likelihood:
        % We adjust the probabilities of transitioning to states consistent vs inconsistent with context.
        % If the context is s1 and face_type is F (0/1), then:
        % - Under context1 (s1=1), P(high tone|sad face)=p_eff, P(low tone|sad face)=1-p_eff; P(low tone|angry)=p_eff, P(high|angry)=1-p_eff.
        % - Under context2 (s1=2), these are inverted.
        % However, since we've explicitly encoded correctness above, the effect of p_eff is implicitly captured by how often the agent expects a trial to align with context or not.
        % In this simplified model, we won't directly apply p_eff to A{1} (feedback) because feedback already reflects actual correctness of this trial.
        % (If simulating the agent's prediction of tone, p_eff would be used, but here each trial's tone is a given event.)
        %
        % [Note: If we wanted to incorporate sensory noise due to intensity, we could soften the entries of A{1} for win/lose based on p_eff, but since each trial is a single observed event, we treat it as given.]
        
        % Fill A{2}: State observation (identity mapping for trial state)
        for s2 = 1:Ns(2)
            % For each trial state, produce the corresponding observation (o = s2) with probability 1 (independent of context)
            A{2}(s2, :, s2) = 1;
        end
        
        % Optionally, create Dirichlet concentration parameters for A (to enable learning A, if desired)
        a{1} = A{1} * times_counts;
        a{2} = A{2} * times_counts;
        
        %% Transition probabilities (B matrices)
        % B{1}: Context transitions (2x2)
        B{1} = [p_stay,   1-p_stay;
                1-p_stay, p_stay];
        
        % B{2}: Trial stage/choice transitions (5x5)
        % Initialize as zero matrix
        B{2} = zeros(Ns(2), Ns(2));
        % From Start state (index 1) to choice states:
        if tone_type == 1
            % High tone presented: allow transitions to states 2 and 3 only.
            % We'll set equal probability for reaching a high-tone state (the agent will then decide via policy which one effectively).
            B{2}(2, 1) = 0.5;
            B{2}(3, 1) = 0.5;
        else
            % Low tone presented: allow transitions to states 4 and 5 only.
            B{2}(4, 1) = 0.5;
            B{2}(5, 1) = 0.5;
        end
        % From any choice state, no further transitions within the trial (absorbing states for that trial).
        B{2}(1, 2:5) = 0;  % (These will be used at next trial as starting at state1 again, handled externally.)
        B{2}(2:5, 2:5) = 0;
        % Note: We leave transitions from choice states to others as 0 because once a choice is made, trial ends (next observation is feedback, then a reset for next trial).
        % The start of the next trial will reset to state1 (Start) via initial conditions rather than B{2} within the same trial.
        
        %% Policies (V matrix)
        % We have 1 action time step (from Start to choice) for factor2. Define policies corresponding to choosing Sad or Angry upon tone observation.
        % Let's define 2 distinct policies:
        % Policy 1: If tone is High -> go to state2 (HighTone-ChooseSad); If tone is Low -> go to state5 (LowTone-ChooseSad)  [i.e., always choose Sad]
        % Policy 2: If tone is High -> go to state3 (HighTone-ChooseAngry); If tone is Low -> go to state4 (LowTone-ChooseAngry)  [i.e., always choose Angry]
        % Represent these in V as sequences of control states for factor2. (Factor1 has no action, so we put 1 as a dummy action for factor1 in all policies.)
        
        Np = 2;  % number of policies
        V = ones(1, Np, Nf);  % initialize (T-1 = 1 action step)
        % Factor1 (context) has no control, so we keep V(:, :, 1) = 1 for all policies (dummy value).
        % Factor2 (trial stage) control: specify which state to transition to from Start.
        % We map: policy1 -> Sad choice, policy2 -> Angry choice.
        if tone_type == 1
            % If high tone this trial, the available transitions from Start are to 2 or 3.
            % But since policies are formulated generally, we set up as if both tone possibilities could occur:
            V(1, 1, 2) = 2;  % Policy1: go to state2 (HighTone-ChooseSad) for high tone case
            V(1, 2, 2) = 3;  % Policy2: go to state3 (HighTone-ChooseAngry)
            % (For low tone, policy1 would go to 5, policy2 to 4. But since only high tone happened, this trial effectively only uses the high-tone branch of policy.)
        else
            % Low tone trial: transitions from Start are to 4 or 5.
            V(1, 1, 2) = 5;  % Policy1: go to state5 (LowTone-ChooseSad)
            V(1, 2, 2) = 4;  % Policy2: go to state4 (LowTone-ChooseAngry)
        end
        % Note: In a full generative model, policies would consider both possible tone outcomes. Here we set V consistent with the actual tone observed on this trial for simplicity.
        
        %% Preferences (C matrices)
        T = 2;  % number of time steps within a trial (observation and then feedback) 
        % (We use T=2 because we consider one observation step for feedback; the initial tone observation is handled implicitly by state transition.)
        C{1} = zeros(No(1), T);
        C{2} = zeros(No(2), T);
        % Set preferences for feedback modality:
        % Time step 1 (no feedback yet) – neutral
        % Time step 2 (feedback received) – dislike lose (-1), like win (+1)
        C{1}(:, 1) = [0; 0; 0];       % at time 1, null/lose/win all 0 preference
        C{1}(:, 2) = [0; -1; 1];      % at time 2, null=0, lose=-1, win=+1
        % No preferences for state observation modality:
        C{2}(:,:) = 0;
        
        %% Populate mdp structure for this trial
        mdp(t).A = A;
        mdp(t).B = B;
        mdp(t).C = C;
        mdp(t).D = D;
        mdp(t).d = d;           % allow learning of context prior
        mdp(t).a = [];          % (we won't explicitly set a here; leaving empty means use fixed A as given. To enable A learning, assign a{1}, a{2} similarly.)
        mdp(t).T = T;
        mdp(t).V = V;
        
        % Save some parameters for reference (not used by spm_MDP_VB, but can be useful for analysis)
        mdp(t).prior_d   = prior_d;
        mdp(t).p_hs_la   = p_hs_la;
        mdp(t).p_correct = p_correct;
        mdp(t).p_stay    = p_stay;
    end
    
    % Check model consistency for all trials
    mdp = spm_MDP_check(mdp);
end