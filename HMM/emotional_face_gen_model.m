function mdp = emotional_face_gen_model(trialinfo,priors)
  
    % priors that we are fixing
    % Feng: this value is the same as "p_right" in the new version of the code
    p_lb = .5; % left better prob in d
    prior_d = 1; % prior d counts
    % Feng: learn_a is a in mdp, not sure what is used for
    learn_a = 1;
    % Feng: this rs is the same as "Rsensitivity" int the prior
    rs = priors.Rsensitivity;
    % Feng: when init the a, it been time outside the matrix, recorad as prior_a in mdp
    prior_a = 1;
    %la = priors.la;
    %eff = priors.eff;

    % Feng: p_ha is the probability of hint accuracy, now called p_a
    % p_ha = priors.p_ha;
    p_ha = priors.p_a;
    % Feng: i guess this alpha is inverse temperature since in the blow comment, it been explained as precision of action
    % alpha = priors.alpha;
    alpha = priors.inv_temp;

    times_counts=200; % orignal version 200



    for t = 1:size(trialinfo,1)
        % Feng, ground truth values for the generative model
        pHA     = str2double(trialinfo{t,1});
        pLB     = str2double(trialinfo{t,2});
        LA      = str2double(trialinfo{t,3})/10;

        % use 4 and 4.7 for loss sizes of 4 and 8 because in exponentiated space
        % 4.7 is double 4
        % feng: not distinguish the loss size here, init it base 
        % [LA , -l_loss_loss * Rsensitivity] for LA == 8
        % [LA , -l_loss_loss] for LA == 4
        % if (LA == 8)
        %     LA = log(exp(4)*2);
        % end

        T = 3;

        % Priors about initial states: D and d

        D{1} = [pLB 1-pLB]';  % {'left better','right better'}
        D{2} = [1 0 0 0]'; % {'start','hint','choose-left','choose-right'}


        d{1} = prior_d*[p_lb 1-p_lb]';  % {'left better','right better'}
        d{2} = [1 0 0 0]'*times_counts; % {'start','hint','choose-left','choose-right'}
        % TODO: maybe could be other value rather than 200, maybe 500 1 etc


        % State-outcome mappings and beliefs: A and a

        Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)

        for i = 1:Ns(2) 

            A{1}(:,:,i) = [1 1; % No Hint
                        0 0; % Machine-Left Hint
                        0 0];% Machine-Right Hint
        end

        % Then we specify that the 'Get Hint' behavior state generates a hint that
        % either the left or right slot machine is better, depending on the context
        % state. In this case, the hints are accurate with a probability of pHA. 

        A{1}(:,:,2) = [0     0;      % No Hint
                    pHA(1) 1-pHA(1);    % Machine-Left Hint
                    1-pHA(1) pHA(1)];   % Machine-Right Hint

        % Next we specify the mapping between states and wins/losses. The first two
        % behavior states ('Start' and 'Get Hint') do not generate either win or
        % loss observations in either context:

        for i = 1:2

            A{2}(:,:,i) = [1 1;  % Null
                        0 0;  % Loss
                        0 0]; % Win
        end
            
        % Choosing the left machine (behavior state 3) generates wins with
        % probability pWin, which differs depending on the context state (columns):



        A{2}(:,:,3) = [0      0;     % Null        
                    0      1;  % Loss
                    1      0]; % Win

        % Choosing the right machine (behavior state 4) generates wins with
        % probability pWin, with the reverse mapping to context states from 
        % choosing the left machine:
                
        A{2}(:,:,4) = [0      0;     % Null
                    1      0;  % Loss
                    0      1]; % Win
                
        % Finally, we specify an identity mapping between behavior states and
        % observed behaviors, to ensure the agent knows that behaviors were carried
        % out as planned. Here, each row corresponds to each behavior state.
                
        for i = 1:Ns(2) 

            A{3}(i,:,i) = [1 1];

        end

        %--------------------------------------------------------------------------
        % Specify prior beliefs about state-outcome mappings in the generative model 
        % (a)
        % Note: This is optional, and will simulate learning state-outcome mappings 
        % if specified.
        %--------------------------------------------------------------------------

        % For example, to simulate learning the reward probabilities, we could specify:
            
        %     a{1} = A{1}*200;
        %     a{2} = A{2}*200;
        %     a{3} = A{3}*200;
        %     
        %     a{2}(:,:,3) =  [0  0;  % Null        
        %                    .5 .5;  % Loss
        %                    .5 .5]; % Win
        %     
        %     
        %     a{2}(:,:,4) =  [0  0;  % Null        
        %                    .5 .5;  % Loss
        %                    .5 .5]; % Win

        % As another example, to simulate learning the hint accuracy one
        % might specify:

        a{1} = A{1}*times_counts;
        a{2} = A{2}*times_counts;
        a{3} = A{3}*times_counts;
        
            
        a{1}(:,:,2) =  [0     0;      % No Hint
                    p_ha 1-p_ha;    % Machine-Left Hint
                    1-p_ha p_ha]*prior_a;   % Machine-Right Hint
                

        B{1}(:,:,1) = [1 0;  % 'Left Better' Context
                    0 1]; % 'Right Better' Context

        % Move to the Start state from any other state
        B{2}(:,:,1) = [1 1 1 1;  % Start State
                    0 0 0 0;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine
                
        % Move to the Hint state from any other state
        B{2}(:,:,2) = [0 0 0 0;  % Start State
                    1 1 1 1;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine

        % Move to the Choose Left state from any other state
        B{2}(:,:,3) = [0 0 0 0;  % Start State
                    0 0 0 0;  % Hint
                    1 1 1 1;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine

        % Move to the Choose Right state from any other state
        B{2}(:,:,4) = [0 0 0 0;  % Start State
                    0 0 0 0;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    1 1 1 1]; % Choose Right Machine        

        % Preferred outcomes: C and c
        % We can start by setting a 0 preference for all outcomes:

        No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % number of outcomes in 
                                                    % each outcome modality

        C{1}      = zeros(No(1),T); % Hints
        C{2}      = zeros(No(2),T); % Wins/Losses
        C{3}      = zeros(No(3),T); % Observed Behaviors

        % Then we can specify a 'loss aversion' magnitude (la) at time points 2 
        % and 3, and a 'reward seeking' (or 'risk-seeking') magnitude (rs). Here,
        % rs is divided by 2 at the third time point to encode a smaller win ($2
        % instead of $4) if taking the hint before choosing a slot machine.

        % C{1}(:,:) =    [0  0   0   ;  % Null
        %                 0 -eff  -eff;  % hint left
        %                 0  -eff  -eff]; % hint right


        %  C{2}(:,:) =    [0      0        0;  % Null
        %                  0 -LA(1)*la -LA(1)*la  ;  % Loss
        %                  0      4        log(exp(4)/2)]; % win
        % Feng: depending on the partsize and time(if take advise) asign the value to C


        C{2}(:,:) =    [0      0        0;  % Null
                 0 -LA(1) -LA(1)  ;  % Loss
                 0      4*rs        log(exp(4)/2)*rs]; % win

        % loss = 0;
        % if (LA == 8)    
        %     loss = priors.l_loss_value*priors.Rsensitivity;
        % elseif (LA == 4)
        %     loss = priors.l_loss_value;
        % end 

        % C{2}(:,:) =    [0       0                               0;  % Null
        %                 0       -loss*priors.reward_value       -loss*priors.reward_value  ;  % Loss
        %                 0       4*priors.reward_value           2*priors.reward_value]; % win


        %                 % T1  T2  T3
        % C{2}(:,:) =    [0      0        0;  % Null
        %                 0 -LA(1) -LA(1)  ;  % Loss
        %                 0      4        2]*rs; % win

        % C{2}(:,:) =    [0  0   0   ;  % Null
        %                 0 -LA(1)*la -LA(1)*la  ;  % Loss
        %                 0  4*rs  2*rs]; % win

        % C{2}(:,:) =    [0  0   0   ;  % Null
        %                 0 -LA(1)*la -LA(1)*la  ;  % Loss
        %                 0  4  2]*pw; % win
        %             
                
        % For our simulations, we will specify V, where rows correspond to time 
        % points and should be length T-1 (here, 2 transitions, from time point 1
        % to time point 2, and time point 2 to time point 3):

        Np = 4; % Number of policies
        Nf = 2; % Number of state factors
        % Feng: What is the meaning of the policy defined here? how to change that?
        V         = ones(T-1,Np,Nf);

        % note that I took out the policy where an agent can take the hint (2) and
        % then do nothing (1) as well as do nothing and do nothing 
        V(:,:,1) = [1 1 1 1;
                    1 1 1 1]; % Context state is not controllable

        V(:,:,2) = [2 2 3 4;
                    3 4 1 1];
                
        % For V(:,:,2), columns left to right indicate policies allowing: 
        % 1. staying in the start state 
        % 2. choosing the hint then returning to start state
        % 3. taking the hint then choosing the left machine
        % 4. taking the hint then choosing the right machine
        % 5. choosing the left machine right away (then returning to start state)
        % 6. choosing the right machine right away (then returning to start state)


        % Additional optional parameters. 
        % Feng: fixed value of those three parameters is reasonable or not.
        beta = 1; % By default this is set to 1, but try increasing its value 
                    % to lower precision and see how it affects model behavior

        erp = 1; % By default we here set this to 1, but try increasing its value  
                % to see how it affects simulated neural (and behavioral) responses
                
        % changed tau to 2; quicker jumps                          
        tau = 2; % Here we set this to 12 to simulate smooth physiological responses,   
                % but try adjusting its value to see how it affects simulated
                    % neural (and behavioral) responses
                    
        mdp(t).T = T;                    % Number of time steps
        mdp(t).V = V;                    % allowable (deep) policies

        %mdp.U = U;                   % We could have instead used shallow 
                                    % policies (specifying U instead of V).

        mdp(t).A = A;                    % state-outcome mapping
        mdp(t).B = B;                    % transition probabilities
        mdp(t).C = C;                    % preferred states
        mdp(t).D = D;                    % priors over initial states

        mdp(t).d = d;                    % enable learning priors over initial states
            
        %mdp(t).eta = eta;                % learning rate
        %mdp(t).eta_win = eta_win;                % learning rate
        %mdp(t).eta_loss = eta_loss;                % learning rate
        % mdp(t).omega_advisor_win = omega_advisor_win;            % forgetting rate
        % mdp(t).omega_advisor_loss = omega_advisor_loss;  
        % mdp(t).omega_context = omega_context;  


        %feng: split the omega and eta values for different states
        mdp(t).omega_d_win = omega_d_win;            % forgetting rate
        mdp(t).omega_d_loss = omega_d_loss;
        mdp(t).omega_a_win = omega_a_win;            % forgetting rate
        mdp(t).omega_a_loss = omega_a_loss;
        mdp(t).eta_d_win = eta_d_win;            % learning rate
        mdp(t).eta_d_loss = eta_d_loss;
        mdp(t).eta_a_win = eta_a_win;            % learning rate
        mdp(t).eta_a_loss = eta_a_loss;
        % mdp(t).omega_eta_advisor_win = omega_eta_advisor_win;            
        % mdp(t).omega_eta_advisor_loss = omega_eta_advisor_loss;  
        % mdp(t).omega_eta_context = omega_eta_context;  

        
        mdp(t).alpha = alpha;            % action precision fixed at 1
        mdp(t).beta = beta;              % expected precision of expected free energy over policies
        mdp(t).erp = erp;                % degree of belief resetting at each timestep
        mdp(t).tau = tau;                % time constant for evidence accumulation
        mdp(t).prior_d = prior_d;
        mdp(t).p_ha = p_ha;
        mdp(t).prior_a = prior_a;
        mdp(t).rs = rs;
        % mdp(t).novelty_scalar = novelty_scalar;
        %mdp(t).la = la;
        %mdp(t).eff = eff;


        % Note, here we are not including habits:
        % or learning other parameters:
        if learn_a ==1
            mdp(t).a = a;   
        end

        mdp(t).a_floor = a{1}(:,:,2);
        mdp(t).d_floor = d{1};

        % We can add labels to states, outcomes, and actions for subsequent plotting:

        label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
        label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
        label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
        label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
        label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
        label.action{2} = {'start','hint','left','right'};
        mdp(t).label = label;
    end
    %--------------------------------------------------------------------------
    % Use a script to check if all matrix-dimensions are correct:
    %--------------------------------------------------------------------------
    mdp = spm_MDP_check(mdp);