function mdp = emotional_face_gen_model(trialinfo,priors)
    p_hs_la = priors.p_hs_la; % probability of high tone given sad face and low tone given angry face
    p_correct = priors.p_correct; % probability of correct response given the correct association
    p_stay = priors.p_stay; % probability of staying in the same state
    prior_d = 1; % prior d counts

    times_counts=200; % orignal version 200

    for t = 1:size(trialinfo,1)

        prob_hightone_sad = trialinfo.prob_hightone_sad(t); % probability of high tone given sad face

        % number of time steps      
        T  = 3;           
        % Priors about initial states: D and d
        % beliefs about the initial state of the context factor sad_hightone-angry_lowtone association or 
        % sad_lowtone-angry_hightone association
        D{1} = [prob_hightone_sad 1-prob_hightone_sad]'; 
        % TODO: should it be 5 states and 3 time steps?
        % TODO: how to add intensity? should face be part of state factor?
        % TODO: should intensity be part of the state factor?
        % D{2} ={start, high tone- choice sad, high tone-choice angry, low tone- choice angry, low tone- choice sad}
        D{2} = [1,0,0,0,0,0,0];
        % intial beliefs about the association
        d{1} = prior_d*[p_hs_la 1-p_hs_la]'; 
        d{2} = [1 0 0 0 0 0 0]';
        Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)
        % likelihood.A
        % col is the state factor(s_h association or s_l association)
        % For Result outcomes Modality
        % TODO: Missing response is not included in this model since RT wont count in HMM model right?
        % rows: Null, Correct, Incorrect
        % start state has no outcome
        A{1}(:,:,1) = [1 1 ; % Null Outcome
                        0 0 ; % Correct Outcome
                        0 0 ]; % Incorrect Outcome
        % high tone  sad state
        A{1}(:,:,2) = [0 0 ; % Null Outcome
                        p_correct 1-p_correct ; % Correct Outcome
                        1-p_correct p_correct ]; % Incorrect Outcome
        % high tone angry state
        A{1}(:,:,3) = [0 0 ; % Null Outcome
                      1-p_correct p_correct ; % Correct Outcome
                      p_correct 1-p_correct ]; % Incorrect Outcome
        % low tone angry state
        A{1}(:,:,4) = [0 0 ; % Null Outcome
                        p_correct 1-p_correct ; % Correct Outcome
                        1-p_correct p_correct ]; % Incorrect Outcome
        % low tone sad state
        A{1}(:,:,5) = [0 0 ; % Null Outcome
                       1-p_correct p_correct ; % Correct Outcome
                       p_correct 1-p_correct ]; % Incorrect Outcome
        
        % For State outcomes Modality
        for i = 1:Ns(2) 
            A{2}(i,:,i) = [1 1];
        end

        a{1} = A{1}*times_counts;

    
       
                
        % transition: B{1}(:,:,1) is a 2×2 matrix, no control inputs for HMM

        B{1}(:,:,1) = [p_stay   1-p_stay;
                    1-p_stay p_stay];
        % Move to the Start state from any other state
        B{2}(:,:,1) = [1 1 1 1;  % Start State
                    0 0 0 0;  % High Tone Choice Sad State
                    0 0 0 0;  % High Tone Choice Angry State
                    0 0 0 0;  % Low Tone Choice Angry State
                    0 0 0 0]; % Low Tone Choice Sad State
        % Move to the High Tone Choice Sad state from any other state
        % TODO!



        No = [size(A{1},1) size(A{2},1)]; % number of outcomes in 
                                                    % each outcome modality
        C{1}      = zeros(No(1),T); % Correct/Incorrect Outcomes
        C{2}      = zeros(No(2),T); % Observed Behaviors

        % Preferred outcomes: C and c
        % zero‐preferences (so model doesn’t “choose” actions)
        % col is time step
        C{1}(:,:) =    [0      0  ;  % Null
            0 -1 ;  % Loss
            0  1 ]; % win


        Np = 5; % Number of policies
        Nf = 2; % Number of state factors
        % Feng: What is the meaning of the policy defined here? how to change that?
        V         = ones(T-1,Np,Nf);

                

        mdp(t).T = T;                    % Number of time steps
        mdp(t).V = V;                    % allowable (deep) policies

        %mdp.U = U;                   % We could have instead used shallow 
                                    % policies (specifying U instead of V).

        mdp(t).A = A;                    % state-outcome mapping
        mdp(t).B = B;                    % transition probabilities
        mdp(t).C = C;                    % preferred states
        mdp(t).D = D;                    % priors over initial states

        mdp(t).d = d;                    % enable learning priors over initial states
       

        
        mdp(t).prior_d = prior_d;
        mdp(t).p_hs_la = p_hs_la;
        mdp(t).p_correct = p_correct;
        mdp(t).p_stay = p_stay;


        % mdp(t).a_floor = a{1}(:,:,2);
        % mdp(t).d_floor = d{1};

        % We can add labels to states, outcomes, and actions for subsequent plotting:

        % label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
        % label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
        % label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
        % label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
        % label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
        % label.action{2} = {'start','hint','left','right'};
        % mdp(t).label = label;
    end
    %--------------------------------------------------------------------------
    % Use a script to check if all matrix-dimensions are correct:
    %--------------------------------------------------------------------------
    mdp = spm_MDP_check(mdp);