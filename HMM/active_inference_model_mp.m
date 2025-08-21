function [MDP] = active_inference_model_mp(task, MDP, params, sim)
    % deal with a sequence of trials
    %==========================================================================

    % options
    %--------------------------------------------------------------------------
    try OPTIONS.plot;  catch, OPTIONS.plot  = 0; end
    try OPTIONS.gamma; catch, OPTIONS.gamma = 1; end
    try OPTIONS.D;     catch, OPTIONS.D     = 0; end
    global MP;
    MP = 0;
    % check MDP specification
    %--------------------------------------------------------------------------
    %MDP = spm_MDP_check(MDP);
    
    % if there are multiple trials ensure that parameters are updated
    %--------------------------------------------------------------------------
    if size(MDP,2) > 1
        
        % plotting options
        %----------------------------------------------------------------------
        GRAPH        = OPTIONS.plot;
        OPTIONS.plot = 0;
        
        for i = 1:size(MDP,2)
            for m = 1:size(MDP,1)
                
                % update concentration parameters
                % as in cary them over from the last trial
                %--------------------------------------------------------------
                if i > 1
                    try,  MDP(m,i).a = OUT(m,i - 1).a; end
                    try,  MDP(m,i).b = OUT(m,i - 1).b; end
                    try,  MDP(m,i).d = OUT(m,i - 1).d; end
                    try,  MDP(m,i).c = OUT(m,i - 1).c; end
                    try,  MDP(m,i).e = OUT(m,i - 1).e; end
                    
                    % update initial states (post-diction)
                    %----------------------------------------------------------
                    if OPTIONS.D
                        for f = 1:numel(MDP(m,i).D)
                            MDP(m,i).D{f} = OUT(m,i - 1).X{f}(:,1);
                        end
                    end
                end
            end
            
            % solve this trial (for all models synchronously)
            %------------------------------------------------------------------
            OUT(:,i) = active_inference_model_mp(task,MDP(:,i),params,sim);
            
            % Bayesian model reduction
            %------------------------------------------------------------------
            if isfield(OPTIONS,'BMR')
                for m = 1:size(MDP,1)
                    OUT(m,i) = spm_MDP_VB_sleep(OUT(m,i),OPTIONS.BMR);
                end
            end
            
        end
        
        % plot summary statistics - over trials
        %----------------------------------------------------------------------
        MDP = OUT;
        if GRAPH
            if ishandle(GRAPH)
                figure(GRAPH); clf
            else
                spm_figure('GetWin','MDP'); clf
            end
            spm_MDP_VB_game(MDP(1,:))
        end
        return
    end
    
    
    %  1. Set up and preliminaries
    %==========================================================================
    % defaults
    %--------------------------------------------------------------------------
    %  Read out all the parameters from the MDP structure, if missing, set to default values
    try, 
        erp   = MDP(1).erp; 
    catch, 
        erp   = 4;    
        % disp('erp is missing, set to default value ');
    end
    try, 
        beta  = MDP(1).beta;  
    catch, 
        beta  = 1;   
        % disp('beta is missing, set to default value '); 
    end

    try, 
        p_hs_la  = MDP(1).p_hs_la;  
    catch, 
        MDP(1).p_hs_la  = 0.5;  
        disp('p_hs_la is missing, set to default value ');
    end
    try, 
        p_high_intensity  = MDP(1).p_high_intensity;  
    catch, 
        MDP(1).p_high_intensity  = 0.7;  
        disp('p_high_intensity is missing, set to default value ');
    end
    try, 
        p_low_intensity  = MDP(1).p_low_intensity;  
    catch, 
        MDP(1).p_low_intensity  = 0.3;  
        disp('p_low_intensity is missing, set to default value ');
    end


    % preclude precision updates for moving policies
    %--------------------------------------------------------------------------
    if isfield(MDP,'U'), OPTIONS.gamma = 1;end
    % Feng: read out the policy amd time from the MDP structure
    for m = 1:size(MDP,1)
        if isfield(MDP(m),'O') && size(MDP(m).U,2) < 2
            % no policies – assume hidden Markov model (HMM)
            %------------------------------------------------------------------
            T(m) = size(MDP(m).O{1},2);         % HMM mode
            V{m} = ones(T - 1,1);               % single 'policy'
            HMM  = 1;
        elseif isfield(MDP(m),'U')
            % called with repeatable actions (U,T)
            %------------------------------------------------------------------
            T(m) = MDP(m).T;                    % number of updates
            V{m} = MDP(m).U;                    % allowable actions (1,Np)
            HMM  = 0;
        elseif isfield(MDP(m),'V')
            % full sequential policies (V)
            %------------------------------------------------------------------
            V{m} = MDP(m).V;                    % allowable policies (T - 1,Np)
            T(m) = size(MDP(m).V,1) + 1;        % number of transitions
            HMM  = 0;
        else
            sprintf('Please specify MDP(%d).U, MDP(%i).V or MDP(%d).O',m), return
        end
    end
    
    % initialise model-specific variables
    %--------------------------------------------------------------------------
    % Read Out and set up initial values for the variables
    % T: number of time steps
    % Ni: number of VB iterations
    % Ng: number of outcome factors
    % Nf: number of hidden state factors
    % Np: number of allowable policies
    % Ns: number of hidden states
    % Nu: number of hidden controls
    % No: number of outcomes
    % A: likelihood model (for a partially observed MDP)
    % pA: prior concentration parameters for complexity (and novelty)
    % wA: normalized importance of the Dirichlet prior parameters a{g}
    % sB: 
    % rB: normailzed b under f(hidden state factors)
    % pB: prior b, unorilzed
    % D: prior over initial hidden states - concentration parameters
    % pD: prior concentration paramters for complexity
    % wD: normalized importance of the Dirichlet prior parameters d{f} but no > 0
    % E: prior over policies - concentration parameters
    % qE: log of E before norm
    % pE: prior concentration paramters for complexity e
    % C: prior preferences (log probabilities)
    % pC: prior preferences

    % inite the posterior expectations of hidden states
    % xn: posterior expectations of hidden states
    % vn: posterior expectations of hidden states
    % x: hidden states
    % X: hidden states
    % p: posterior expectations of hidden states
    % un: posterior expectations of hidden states
    % u: posterior expectations of hidden states



    T     = T(1);                              % number of time steps
    Ni    = 8; 
    if MP == 0
        Ni = 1;
    end
    % number of VB iterations
    for m = 1:size(MDP,1)
        
        % ensure policy length is less than the number of updates
        %----------------------------------------------------------------------
        if size(V{m},1) > (T - 1)
            V{m} = V{m}(1:(T - 1),:,:);
        end
        
        % numbers of transitions, policies and states
        %----------------------------------------------------------------------
        Ng(m) = numel(MDP(m).A);               % number of outcome factors
        Nf(m) = numel(MDP(m).B);               % number of hidden state factors
        Np(m) = size(V{m},2);                  % number of allowable policies
        for f = 1:Nf(m)
            Ns(m,f) = size(MDP(m).B{f},1);     % number of hidden states
            Nu(m,f) = size(MDP(m).B{f},3);     % number of hidden controls
        end
        for g = 1:Ng(m)
            No(m,g) = size(MDP(m).A{g},1);     % number of outcomes
        end
        % parameters of generative model and policies
        %======================================================================
        % likelihood model (for a partially observed MDP)
        %----------------------------------------------------------------------
        for g = 1:Ng(m)
            
            % ensure probabilities are normalised  : A
            %------------------------------------------------------------------
            MDP(m).A{g} = spm_norm(MDP(m).A{g});
            
            % parameters (concentration parameters): a
            %------------------------------------------------------------------
            if isfield(MDP,'a')
                A{m,g}  = spm_norm(MDP(m).a{g});
            else
                A{m,g}  = spm_norm(MDP(m).A{g});
            end
            
            % prior concentration parameters for complexity (and novelty)
            %------------------------------------------------------------------
            if isfield(MDP,'a')
                pA{m,g} = MDP(m).a{g};
                wA{m,g} = spm_wnorm(MDP(m).a{g}).*(pA{m,g} > 0);
            end
            
        end
        
        % transition probabilities (priors)
        %----------------------------------------------------------------------
        for f = 1:Nf(m)
            for j = 1:Nu(m,f)
                % controlable transition probabilities : B
                %--------------------------------------------------------------
                MDP(m).B{f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j));
                % parameters (concentration parameters): b
                %--------------------------------------------------------------
                if isfield(MDP,'b') && ~HMM
                    sB{m,f}(:,:,j) = spm_norm(MDP(m).b{f}(:,:,j) );
                    rB{m,f}(:,:,j) = spm_norm(MDP(m).b{f}(:,:,j)');
                else
                    sB{m,f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j) );
                    rB{m,f}(:,:,j) = spm_norm(MDP(m).B{f}(:,:,j)');
                end
                
            end
            
            % prior concentration paramters for complexity
            %------------------------------------------------------------------
            if isfield(MDP,'b')
                pB{m,f} = MDP(m).b{f};
            end
            
        end
        
        % priors over initial hidden states - concentration parameters
        %----------------------------------------------------------------------
        for f = 1:Nf(m)
            if isfield(MDP,'d')
                D{m,f} = spm_norm(MDP(m).d{f});
            elseif isfield(MDP,'D')
                D{m,f} = spm_norm(MDP(m).D{f});
            else
                D{m,f} = spm_norm(ones(Ns(m,f),1));
                MDP(m).D{f} = D{m,f};
            end
            
            % prior concentration paramters for complexity
            %------------------------------------------------------------------
            if isfield(MDP,'d')
                pD{m,f} = MDP(m).d{f};
                wD{m,f} = spm_wnorm(MDP(m).d{f});
            end
        end
        
        % priors over policies - concentration parameters
        %----------------------------------------------------------------------
        if isfield(MDP,'e')
            E{m} = spm_norm(MDP(m).e);
        elseif isfield(MDP,'E')
            E{m} = spm_norm(MDP(m).E);
        else
            E{m} = spm_norm(ones(Np(m),1));
        end
        qE{m}    = spm_log(E{m});
        
        % prior concentration paramters for complexity
        %----------------------------------------------------------------------
        if isfield(MDP,'e')
            pE{m} = MDP(m).e;
        end
        
        % prior preferences (log probabilities) : C
        %----------------------------------------------------------------------
        for g = 1:Ng(m)
            if isfield(MDP,'c')
                C{m,g}  = spm_psi(MDP(m).c{g} + 1/32);
                pC{m,g} = MDP(m).c{g};
            elseif isfield(MDP,'C')
                C{m,g}  = MDP(m).C{g};
            else
                C{m,g}  = zeros(No(m,g),1);
            end
            
            % assume time-invariant preferences, if unspecified
            %------------------------------------------------------------------
            if size(C{m,g},2) == 1
                C{m,g} = repmat(C{m,g},1,T);
                if isfield(MDP,'c')
                    MDP(m).c{g} = repmat(MDP(m).c{g},1,T);
                    pC{m,g}     = repmat(pC{m,g},1,T);
                end
            end
            C{m,g} = spm_log(spm_softmax(C{m,g}));
        end
        
        % initialise  posterior expectations of hidden states
        %----------------------------------------------------------------------
        for f = 1:Nf(m)
            xn{m,f} = zeros(Ni,Ns(m,f),1,T,Np(m)) + 1/Ns(m,f);
            vn{m,f} = zeros(Ni,Ns(m,f),1,T,Np(m));
            x{m,f}  = zeros(Ns(m,f),T,Np(m))      + 1/Ns(m,f);
            X{m,f}  = repmat(D{m,f},1,1);
            for k = 1:Np(m)
                x{m,f}(:,1,k) = D{m,f};
            end
        end
        
        % initialise posteriors over polices and action
        %----------------------------------------------------------------------
        P{m}  = zeros([Nu(m,:),1]);
        un{m} = zeros(Np(m),1);
        u{m}  = zeros(Np(m),1);
        
        % if there is only one policy
        %----------------------------------------------------------------------
        if Np(m) == 1
            u{m} = ones(Np(m),T);
        end
        
        % if states have not been specified set to 0
        %----------------------------------------------------------------------
        s{m}  = zeros(Nf(m),T);
        % check if already have states, if so, use them
        try
            i = find(MDP(m).s);
            s{m}(i) = MDP(m).s(i);
        end
        MDP(m).s = s{m};
        
        % if outcomes have not been specified set to 0
        %----------------------------------------------------------------------
        % check if already have outcomes, if so, use them
        o{m}  = zeros(Ng(m),T);
        try
            i = find(MDP(m).o);
            o{m}(i) = MDP(m).o(i);
        end
        MDP(m).o = o{m};
        % (indices of) plausible (allowable) policies
        %----------------------------------------------------------------------
        p{m}  = 1:Np(m);
        % expected rate parameter (precision of posterior over policies)
        %----------------------------------------------------------------------
        qb{m} = beta;                          % initialise rate parameters
        w{m}  = 1/qb{m};                       % posterior precision (policy)
        
    end
    % belief updating for shared states and outcomes (multiple models)
    %==========================================================================
    % ensure any outcome generating agent is updated first
    %--------------------------------------------------------------------------
    for m = 1:size(MDP,1)
        n      = -MDP(m).o;
        N(m,:) = mode(n.*(n > 0),1);
    end
    n     = mode(N,1);
    for t = 1:T
        if n(t) > 0
            M(t,:) = circshift((1:size(MDP,1)),[0 (1 - n(t))]);
        else
            M(t,:) = 1;
        end
    end


    % 2. Belief updating over successive time points
    % Feng: almost no change under this part
    %==========================================================================
    for t = 1:T
        if MP == 0
            bound = 1;
        else
            bound = T;
        end
        % generate hidden states and outcomes for each agent or model
        %======================================================================
        for m = M(t,:)
            % if ~HMM % not required for HMM
            %     %  2.1 sample state, if not specified
            %     %--------------------------------------------------------------
            %     for f = 1:Nf(m)
            %         % the next state is generated by action
            %         %----------------------------------------------------------
            %         % if state is not specified(0), sample from transition
            %         if MDP(m).s(f,t) == 0
            %             % use previous state and action to generate next state if the time is not 1
            %             if t > 1
            %                 ps = MDP(m).B{f}(:,MDP(m).s(f,t - 1),MDP(m).u(f,t - 1));
            %             % use initial state to generate next state if the time is 1
            %             else
            %                 ps = spm_norm(MDP(m).D{f});
            %             end
            %             % sample from transition as the next state
            %             MDP(m).s(f,t) = find(rand < cumsum(ps),1);
            %         end
            %     end
            %     %  2.2 posterior predictive density, base on belief of current state and action
            %     %--------------------------------------------------------------
            %     for f = 1:Nf(m)
            %         % under selected action (xqq)
            %         %----------------------------------------------------------
            %         % use the transition matrix to generate the next state if the time is not 1
            %         if t > 1
            %             xqq{m,f} = sB{m,f}(:,:,MDP(m).u(f,t - 1))*X{m,f}(:,t - 1);
            %         % just use current state if the time is 1
            %         else
            %             xqq{m,f} = X{m,f}(:,t);
            %         end
            % 
            %         % Bayesian model average (xq)
            %         %----------------------------------------------------------
            %         xq{m,f} = X{m,f}(:,t);
            %     end
            %     %  2.3 sample outcome, if not specified
            %     % feng: should the outcome already be known here?
            %     %-------------------------------------------------------------
            %     for g = 1:Ng(m)
            %         if MDP(m).o(g,t) < 0
            % 
            %             % outcome is generated by model n
            %             %------------------------------------------------------
            %             n = -MDP(m).o(g,t);
            %             MDP(m).n(g,t) = n;
            %             if n == m
            % 
            %                 % outcome that minimises expected free energy
            %                 %--------------------------------------------------
            %                 po    = spm_dot(A{m,g},xqq(m,:));
            %                 px    = spm_vec(spm_cross(xqq(m,:)));
            %                 F     = zeros(No(m,g),1);
            %                 for i = 1:No(m,g)
            %                     xp   = MDP(m).A{g}(i,:);
            %                     xp   = spm_norm(spm_vec(xp));
            %                     F(i) = spm_vec(px)'*spm_log(xp) + spm_log(po(i));
            %                 end
            %                 po            = spm_softmax(F*512);
            %                 MDP(m).o(g,t) = find(rand < cumsum(po),1);
            % 
            %             else
            % 
            %                 % outcome from model n
            %                 %--------------------------------------------------
            %                 MDP(m).o(g,t) = MDP(n).o(g,t);
            % 
            %             end
            % 
            %         elseif MDP(m).o(g,t) == 0
            % 
            %             % sample outcome from the generative process
            %             %------------------------------------------------------
            %             ind           = num2cell(MDP(m).s(:,t));
            %             po            = MDP(m).A{g}(:,ind{:});
            %             MDP(m).o(g,t) = find(rand < cumsum(po),1);
            %         end
            %     end
            % end % HMM
            
            % get probabilistic outcomes from samples or subordinate level
            %==================================================================
            
            % 2.4 get outcome likelihood (O)
            %------------------------------------------------------------------
            for g = 1:Ng(m)
                % specified as a likelihood or observation (HMM)
                %--------------------------------------------------------------
                if HMM
                    O{m}{g,t} = MDP(m).O{g}(:,t);
                else
                    O{m}{g,t} = sparse(MDP(m).o(g,t),1,1,No(m,g),1);
                end
            end
            
            % 2.5 likelihood (for multiple modalities)
            %==================================================================
            L{m,t} = 1;
            for g = 1:Ng(m)
                L{m,t} = L{m,t}.*spm_dot(A{m,g},O{m}{g,t});
            end
            
            
            % 2.6 Variational updates (skip to t = T in HMM mode)
            %==================================================================
            if ~HMM || T == t
                %  2.6.1 eliminate unlikely policies
                %--------------------------------------------------------------
                if ~isfield(MDP,'U') && t > 1
                    F    = log(u{m}(p{m},t - 1));
                    p{m} = p{m}((F - max(F)) > -zeta);
                end
                
                % 2.6.2 processing time and reset
                %--------------------------------------------------------------
                tstart = tic;
                for f = 1:Nf(m)
                    x{m,f} = spm_softmax(spm_log(x{m,f})/erp);
                    % TODO: read the orgianl doc shared by Carter, if not solved, ask Ryan
                end
                
                % Variational updates (hidden states) under sequential policies
                %==============================================================
                % variational message passing (VMP)
                %--------------------------------------------------------------
                %  number of time steps the model will look ahead.
                S = size(V{m},1) + 1;   % horizon
                % R: The number of future time steps the model will consider when doing backward messages
                if isfield(MDP,'U')
                    R = t;
                else
                    R = S;
                end

                F = zeros(Np(m),1);
                % 2.6. 3loop over plausible policies
                for k = p{m}                % loop over plausible policies
                    % dF is used to track the change in the free energy (F) as we update the beliefs.
                    dF    = 1;              % reset criterion for this policy
                    for i = 1:Ni            % iterate belief updates
                        F(k)  = 0;  
                        % if t == 1 
                        %     bound = T;
                        % elseif MP == 0
                        %     bound = 1;
                        % else
                        %     bound = T;
                        % end
                        % reset free energy for this policy
                        for j = 1:bound         % loop over future time points
                            if MP == 0
                                j = t;
                            end
                            % curent posterior over outcome factors
                            %--------------------------------------------------
                            if j <= t
                                for f = 1:Nf(m)
                                    xq{m,f} = full(x{m,f}(:,j,k));
                                end
                            end
                            % loop over outcome factors
                            for f = 1:Nf(m)
                                likelihoods_Nf = cellfun(@(cell) cell(:, f), L, 'UniformOutput', false);
                                % hidden states for this time and policy
                                %----------------------------------------------
                                sx = full(x{m,f}(:,j,k));
                                qL = zeros(Ns(m,f),1);
                                v  = zeros(Ns(m,f),1);
                                
                                % evaluate free energy and gradients (v = dFdx)
                                %----------------------------------------------
                                if dF > exp(-8) || i > 4
                                    
                                    % marginal likelihood over outcome factors
                                    %------------------------------------------
                                    if j <= t
                                        qL = spm_dot(L{m,j},xq(m,:),f);
                                        LLH{f,t} = qL(:);
                                        qL = spm_log(qL(:));
    
                                    end
                                    
                                    % entropy
                                    %------------------------------------------
                                    qx  = spm_log(sx);
                                    
                                    % emprical priors (forward messages)
                                    %------------------------------------------
                                    if j < 2
                                        px_forward = spm_log(D{m,f});
                                        v  = v + px_forward + qL - qx;
                                    else
                                        % px is prior based on transition
                                        px_forward = spm_log(sB{m,f}(:,:,V{m}(j - 1,k,f))*x{m,f}(:,j - 1,k));
                                        v  = v + px_forward + qL - qx;
                                    end
                                    
                                    % emprical priors (backward messages)
                                    %------------------------------------------
                                    if j < R && MP == 1
                                        px_backward = spm_log(rB{m,f}(:,:,V{m}(j,k,f))*x{m,f}(:,j + 1,k));
                                        v  = v + px_backward + qL - qx;
                                        coef = 0.5;
                                    else
                                        px_backward = 0;
                                        coef = 1;
                                    end
                                    
                                    % (negative) expected free energy
                                    %------------------------------------------
                                    F(k) = F(k) + sx'*v/Nf(m);
                                    
                                    % update
                                    %------------------------------------------
                                    %v    = v - mean(v);
                                    % 1-step bayesian inference still using
                                    % forward and backward messages
                                    
                                    posterior = coef*px_forward + coef*px_backward + qL;
                                    
                                    sx   = spm_softmax(posterior);
                                    if t > 1 && MP == 0
                                        %spm_backwards(O,Q,A,B,u,t,T)
                                        
                                        full_posterior = spm_backwards(O,sx,L(f,:),sB{m,f},V{m}(:,k,f),1,t);
                                        
                                    else
                                        full_posterior = sx;
                                    end
                                else
                                    F(k) = G(k);
                                end
                                
                                % store update neuronal activity
                                %----------------------------------------------
                                if t > 1 && MP == 0
                                    for tstep = 1:length(full_posterior(1,:))
                                        x{m,f}(:,tstep,k) = full_posterior{tstep}(1,:);
                                        xq{m,f}            = full_posterior{tstep}(1,:);
                                        xn{m,f}(1,:,1,tstep,k) = full_posterior{tstep}(1,:);
                                        vn{m,f}(1,:,1,tstep,k) = v;
                                    end
                                end
                                if MP == 0
                                    x{m,f}(:,t,k)  = sx;
                                    xq{m,f} = sx;
                                    xn{m,f}(1,:,1,t,k) = sx;
                                    vn{m,f}(1,:,1,t,k) = v;
                                else
                                    x{m,f}(:,j,k)  = sx;
                                    xq{m,f} = sx;
                                    xn{m,f}(1,:,j,t,k) = sx;
                                    vn{m,f}(1,:,j,t,k) = v;
                                end
                                
                            end
                        end
                        
                        % convergence
                        %------------------------------------------------------
                        if i > 1
                            dF = F(k) - G(k);
                        end
                        G = F;
                        
                    end
                end
                
                % accumulate expected free energy of policies (Q)
                %==============================================================
                pu  = 1;                               % empirical prior
                qu  = 1;                               % posterior
                Q   = zeros(Np(m),1);                  % expected free energy
                if Np(m) > 1
                    for k = p{m}
                        
                        % Bayesian surprise about inital conditions
                        %------------------------------------------------------
                        % novelty for context
                        if isfield(MDP,'d')
                            for f = 1:Nf(m)
                                Q(k) = Q(k) - spm_dot(wD{m,f},x{m,f}(:,1,k));
                            end
                        end
                        
                        for j = t:S
                            
                            % get expected states for this policy and time
                            %--------------------------------------------------
                            for f = 1:Nf(m)
                                if j > 1 && MP == 0
                                    xq{m,f} = sB{m,f}(:,:,V{m}(j - 1,k,f))*x{m,f}(:,j - 1,k);
                                else
                                    xq{m,f} = x{m,f}(:,j,k);
                                end
                            end
                            
                            % (negative) expected free energy
                            %==================================================
                            
                            % Bayesian surprise about states
                            %--------------------------------------------------
                            % epistemic term
                            Q(k) = Q(k) + spm_MDP_G(A(m,:),xq(m,:));
                            
                            for g = 1:Ng(m)
                                
                                % prior preferences about outcomes
                                %----------------------------------------------
                                qo   = spm_dot(A{m,g},xq(m,:));
                                Q(k) = Q(k) + qo'*(C{m,g}(:,j));
                                
                                % Bayesian surprise about parameters
                                %----------------------------------------------
                                % novelty for advisor
                                % ARTIFICIALLY SCALED DOWN BY paramater
                                if isfield(MDP,'a')
                                    Q(k) = Q(k) - novelty_scalar*spm_dot(wA{m,g},{qo xq{m,:}});
                                end
                            end
                        end
                    end
                    
                    
                    % variational updates - policies and precision
                    %==========================================================
                    
                    % previous expected precision
                    %----------------------------------------------------------
                    if t > 1
                        w{m}(t) = w{m}(t - 1);
                        num_columns = Np(m);
                        for j = 1:Nf(m)                       
                            previous_actions = MDP(m).u(j,1:t-1);
                            for i = 1:num_columns
                                mask = isequal(V{m}(1:t-1,i,j),previous_actions');
                                if mask == 0
                                    Q(i) = intmin('int32');
                                end
                            end
                        end
        
                    end
                    p{m} = find(Q > intmin('int32'))';
                    for i = 1:1
                    
                        % posterior and prior beliefs about policies
                        %------------------------------------------------------
                        qu = spm_softmax(qE{m} + (1/beta)*Q);
                        pu = spm_softmax(qE{m} + (1/beta)*Q);
                        
                        % precision (w) with free energy gradients (v = -dF/dw)
                        %------------------------------------------------------
                        if OPTIONS.gamma
                            w{m}(t) = 1/beta;
                        else
                            eg      = (qu - pu)'*Q(p{m});
                            dFdg    = qb{m} - beta + eg;
                            qb{m}   = qb{m} - dFdg/2;
                            w{m}(t) = 1/qb{m};
                        end
                        
                        % simulated dopamine responses (expected precision)
                        %------------------------------------------------------
                        n             = (t - 1)*Ni + i;
                        wn{m}(n,1)    = w{m}(t);
                        un{m}(:,n) = qu;
                        u{m}(:,t)  = qu;
                        
                    end               
                end % end of loop over multiple policies
                
                
                % Bayesian model averaging of hidden states (over policies)
                %--------------------------------------------------------------
                for f = 1:Nf(m)
                    for i = 1:S
                        X{m,f}(:,i) = reshape(x{m,f}(:,i,:),Ns(m,f),Np(m))*u{m}(:,t);
                    end
                end
                
                % processing (i.e., reaction) time
                %--------------------------------------------------------------
                rt{m}(t)      = toc(tstart);
                
                % record (negative) free energies
                %--------------------------------------------------------------
                MDP(m).F(:,t) = F;
                MDP(m).G(:,t) = Q;
                MDP(m).H(1,t) = 1;%qu'*MDP(m).F(p{m},t) - qu'*(log(qu) - log(pu));
                
                % check for residual uncertainty (in hierarchical schemes)
                %--------------------------------------------------------------
                if isfield(MDP,'factor')
                    
                    for f = MDP(m).factor(:)'
                        qx     = X{m,f}(:,1);
                        H(m,f) = qx'*spm_log(qx);
                    end
                    
                    % break if there is no further uncertainty to resolve
                    %----------------------------------------------------------
                    if sum(H(:)) > - chi
                        T = t;
                    end
                end
                
                
                % action selection
                %==============================================================
                if t < T
                    
                    % marginal posterior over action (for each modality)
                    %----------------------------------------------------------
                    Pu    = zeros([Nu(m,:),1]);
                    for i = 1:Np(m)
                        sub        = num2cell(V{m}(t,i,:));
                        % instead of adding probs changed to taking max of
                        % probs
                        %Pu(sub{:}) = Pu(sub{:}) + u{m}(i,t);
                        Pu(sub{:}) = max(Pu(sub{:}),u{m}(i,t));
                    end
                    
                    % action selection (softmax function of action potential)
                    %----------------------`------------------------------------
                    sub            = repmat({':'},1,Nf(m));
                    Pu(:)          = spm_softmax(alpha*log(Pu(:)));
                    P{m}(sub{:},t) = Pu;
                    
                    % next action - sampled from marginal posterior
                    %----------------------------------------------------------
                    try
                        MDP(m).u(:,t) = MDP(m).u(:,t);
                    catch
                        ind           = find(rand < cumsum(Pu(:)),1);
                        MDP(m).u(:,t) = spm_ind2sub(Nu(m,:),ind);
                    end
                    
                    % update policy and states for moving policies
                    %----------------------------------------------------------
                    if isfield(MDP,'U')
                        
                        for f = 1:Nf(m)
                            V{m}(t,:,f) = MDP(m).u(f,t);
                        end
                        for j = 1:size(MDP(m).U,1)
                            if (t + j) < T
                                V{m}(t + j,:,:) = MDP(m).U(j,:,:);
                            end
                        end
                        
                        % and re-initialise expectations about hidden states
                        %------------------------------------------------------
                        for f = 1:Nf(m)
                            for k = 1:Np(m)
                                x{m,f}(:,:,k) = 1/Ns(m,f);
                            end
                        end
                    end
                    
                end % end of state and action selection
            end % end of variational updates over time

        end % end of loop over models (agents)
        
        % 2.7 terminate evidence accumulation
        %----------------------------------------------------------------------
        if t == T
            if T == 1
                MDP(m).u = zeros(Nf(m),0);
            end
            if ~HMM
                MDP(m).o  = MDP(m).o(:,1:T);        % outcomes at 1,...,T
                MDP(m).s  = MDP(m).s(:,1:T);        % states   at 1,...,T
                MDP(m).u  = MDP(m).u(:,1:T - 1);    % actions  at 1,...,T - 1
            end
            break;
        end
        
    end % end of loop over time
    

    





    % 3. learning – accumulate concentration parameters
    %==========================================================================
    for m = 1:size(MDP,1)
        for t = 2
            % mapping from hidden states to outcomes: a
            if isfield(MDP,'a')
                for g = 1
                    da     = O{m}(g,t);
                    for  f = 1:Nf(m)
                        da = spm_cross(da,X{m,f}(:,t));
                    end
                    da     = da.*(MDP(m).a{g} > 0);
                    %                 MDP(m).a{g}(:,:,2) = MDP(m).a{g}(:,:,2)*omega + da(:,:,2)*eta;
                    %                MDP(m).a{g}(:,:,2) = (MDP(m).a{g}(:,:,2) - MDP.a_floor(:,:))*omega + MDP.a_floor(:,:) +da(:,:,2)*eta;
                    % apply sep learning and forgetting rates
                    % update concentration matrix diff for wins and losses
                    % first determine if it was a win or loss by getting result at t=2
                    % and t=3
                    %                 result = (O{m}(2,2));
                    %                 t_two_result = find(result{:});
                    %                 result = (O{m}(2,3));
                    %                 t_three_result = find(result{:});  

                    result = (O{m}(2,2));
                    t_two_result = find(result{:});
                    result = (O{m}(2,3));
                    t_three_result = find(result{:}); 
                    result = (O{m}(1,2));
                    advisor_chosen = find(result{:});

                    
                    %  only learn/forget when the advisor was chosen (i.e. hint left or hint right). 
                    % apply omega_advisor_loss and omega_advisor_win at the
                    % same time
                    if (advisor_chosen == 2 | advisor_chosen == 3)
                        a_learned = (MDP(m).a{g}(:,:,2) - MDP.a_floor(:,:));
                        % Feng:
                        if (t_two_result == 3 | t_three_result == 3)
                            MDP(m).a{g}(:,:,2) = a_learned * omega_a_win + MDP.a_floor(:,:) +da(:,:,2)*eta_a_win;
                        else
                            MDP(m).a{g}(:,:,2) = a_learned * omega_a_loss + MDP.a_floor(:,:) +da(:,:,2)*eta_a_loss;
                        end
                    end
                    %                 for q = 1:2
                    %                      for r = 2:3
                    %                         if MDP(m).a{g}(r,q,2) < MDP.a_floor(r,q)
                    %                             MDP(m).a{g}(r,q,2) = MDP.a_floor(r,q);
                    %                         end
                    %                      end
                    %                  end
                end
            end
            % mapping from hidden states to hidden states: b(u)
            %------------------------------------------------------------------
            if isfield(MDP,'b') && t > 1
                for f = 1:Nf(m)
                    for k = 1:Np(m)
                        v   = V{m}(t - 1,k,f);
                        db  = u{m}(k,t)*x{m,f}(:,t,k)*x{m,f}(:,t - 1,k)';
                        db  = db.*(MDP(m).b{f}(:,:,v) > 0);
                        MDP(m).b{f}(:,:,v) = MDP(m).b{f}(:,:,v)*omega + db*eta;
                    end
                end
            end
                    
                    % accumulation of prior preferences: (c)
            %------------------------------------------------------------------
            if isfield(MDP,'c')
                for g = 1:Ng(m)
                    dc = O{m}(g,t);
                    if size(MDP(m).c{g},2) > 1
                        dc = dc.*(MDP(m).c{g}(:,t) > 0);
                        MDP(m).c{g}(:,t) = MDP(m).c{g}(:,t) + dc*eta;
                    else
                        dc = dc.*(MDP(m).c{g}>0);
                        MDP(m).c{g} = MDP(m).c{g}*omega + dc*eta;
                    end
                end
            end
        end
                
        % initial hidden states:
        if isfield(MDP,'d')
            for f = 1
                i = MDP(m).d{f} > 0;

                % why are we using the posterior over states at time t=1 to
                % update concentration param for context
                %MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*omega + MDP(m).d_floor + X{m,f}(i,1)*eta;
                % update belief at time 3
                %MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*omega_context + MDP(m).d_floor + X{m,f}(i,3)*eta;
                
                
                % Feng:
                % Update the initial hidden states with the context learning rate and forgeting rate
     
                
                % 
                % if advisor_chosen == 1
                %     choose_left = O{m}(3,2);
                % else
                %     choose_left = O{m}(3,3);
                % end
                % 
                % result = (O{m}(2,2));
                % t_two_result = find(result{:});
                % result = (O{m}(2,3));
                % t_three_result = find(result{:}); 
                % result = (O{m}(1,2));
                % advisor_chosen = find(result{:});

                
             
          
                % if (t_two_result == 3 | t_three_result == 3)
                % 
                %     MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*eta_d_win + MDP(m).d_floor + X{m,f}(i,3)*(1-omega_d_win);
                %     % MDP(m).d{f}(1) = (MDP(m).d{f}(1) - MDP(m).d_floor(1))*eta_d_win + MDP(m).d_floor(1) + (tmp(1))*(1-omega_d_win);
                %     % MDP(m).d{f}(2) = (MDP(m).d{f}(2) - MDP(m).d_floor(2))*eta_d_loss + MDP(m).d_floor(2) + (tmp(2))*(1-omega_d_loss);
                % else
                % % if choose right and win, left and lose
                %     MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*eta_d_loss + MDP(m).d_floor + X{m,f}(i,3)*(1-omega_d_loss);
                %     % MDP(m).d{f}(1) = (MDP(m).d{f}(1) - MDP(m).d_floor(1))*eta_d_loss + MDP(m).d_floor(1) + (tmp(1))*(1-omega_d_loss);
                %     % MDP(m).d{f}(2) = (MDP(m).d{f}(2) - MDP(m).d_floor(2))*eta_d_win + MDP(m).d_floor(2) + (tmp(2))*(1-omega_d_win);
                % end

                MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*eta_d_loss + MDP(m).d_floor + X{m,f}(i,3)*(1-omega_d_loss);
                % MDP(m).d{f} = (MDP(m).d{f} - MDP(m).d_floor)*omega_eta_context + MDP(m).d_floor + X{m,f}(i,3)*(1-omega_eta_context);
                

                %             MDP(m).d{f}(i) = MDP(m).d{f}(i)*omega + X{m,f}(i,1)*eta;
                %                 for q = 1:2
                %                     if MDP(m).d{f}(q) < MDP.d_floor(q)
                %                        MDP(m).d{f}(q) = MDP.d_floor(q); 
                %                     end
                %                 end
            end
        end
        % policies
        %----------------------------------------------------------------------
        if isfield(MDP,'e')
            MDP(m).e = MDP(m).e*omega + eta*u{m}(:,T);
        end
        % (negative) free energy of parameters (complexity): outcome specific
        %----------------------------------------------------------------------
        for g = 1:Ng(m)
            if isfield(MDP,'a')
                MDP(m).Fa(g) = - spm_KL_dir(MDP(m).a{g},pA{m,g});
            end
            if isfield(MDP,'c')
                MDP(m).Fc(f) = - spm_KL_dir(MDP(m).c{g},pC{g});
            end
        end
        
        % (negative) free energy of parameters: state specific
        %----------------------------------------------------------------------
        for f = 1:Nf(m)
            if isfield(MDP,'b')
                MDP(m).Fb(f) = - spm_KL_dir(MDP(m).b{f},pB{m,f});
            end
            if isfield(MDP,'d')
                MDP(m).Fd(f) = - spm_KL_dir(MDP(m).d{f},pD{m,f});
            end
        end
        
        % (negative) free energy of parameters: policy specific
        %----------------------------------------------------------------------
        if isfield(MDP,'e')
            MDP(m).Fe = - spm_KL_dir(MDP(m).e,pE{m});
        end
        
        % simulated dopamine (or cholinergic) responses
        %----------------------------------------------------------------------
        if Np(m) > 1
            dn{m} = 8*gradient(wn{m}) + wn{m}/8;
        else
            dn{m} = [];
            wn{m} = [];
        end
        
        % Bayesian model averaging of expected hidden states over policies
        %----------------------------------------------------------------------
        for f = 1:Nf(m)
            Xn{m,f} = zeros(Ni,Ns(m,f),T,T);
            Vn{m,f} = zeros(Ni,Ns(m,f),T,T);
            if MP == 0
                for i = 1:T
                    for k = 1:Np(m)
                        Xn{m,f}(:,:,:,i) = Xn{m,f}(:,:,:,i) + xn{m,f}(:,:,1,i,k)*u{m}(k,i);
                        Vn{m,f}(:,:,:,i) = Vn{m,f}(:,:,:,i) + vn{m,f}(:,:,1,i,k)*u{m}(k,i);
                    end
                end
            else
                for i = 1:T
                    for k = 1:Np(m)
                        Xn{m,f}(:,:,:,i) = Xn{m,f}(:,:,:,i) + xn{m,f}(:,:,1:T,i,k)*u{m}(k,i);
                        Vn{m,f}(:,:,:,i) = Vn{m,f}(:,:,:,i) + vn{m,f}(:,:,1:T,i,k)*u{m}(k,i);
                    end
                end
            end
        end
        
        % use penultimate beliefs about moving policies
        %----------------------------------------------------------------------
        if isfield(MDP,'U')
            u{m}(:,T)  = [];
            try un{m}(:,(end - Ni + 1):end) = []; catch, end
        end
        
        % assemble results and place in NDP structure
        %----------------------------------------------------------------------
        MDP(m).T  = T;            % number of belief updates
        MDP(m).V  = V{m};         % policies
        MDP(m).O  = O{m};         % policies
        MDP(m).P  = P{m};         % probability of action at time 1,...,T - 1
        MDP(m).R  = u{m};         % conditional expectations over policies
        MDP(m).Q  = x(m,:);       % conditional expectations over N states
        MDP(m).X  = X(m,:);       % Bayesian model averages over T outcomes
        MDP(m).C  = C(m,:);       % utility
        
        if HMM
            MDP(m).o  = zeros(Ng(m),0);      % outcomes at 1,...,T
            MDP(m).s  = zeros(Nf(m),0);      % states   at 1,...,T
            MDP(m).u  = zeros(Nf(m),0);      % actions  at 1,...,T - 1
            return
        end
        
        MDP(m).w  = w{m};         % posterior expectations of precision (policy)
        MDP(m).vn = Vn(m,:);      % simulated neuronal prediction error
        MDP(m).xn = Xn(m,:);      % simulated neuronal encoding of hidden states
        MDP(m).un = un{m};        % simulated neuronal encoding of policies
        MDP(m).wn = wn{m};        % simulated neuronal encoding of precision
        MDP(m).dn = dn{m};        % simulated dopamine responses (deconvolved)
        MDP(m).rt = rt{m};        % simulated reaction time (seconds)
        
    end
    
    
    % plot
    %==========================================================================
    if OPTIONS.plot
        if ishandle(OPTIONS.plot)
            figure(OPTIONS.plot); clf
        else
            spm_figure('GetWin','MDP'); clf
        end
        spm_MDP_VB_trial(MDP(1))
    end
end
        

% auxillary functions
%==========================================================================

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);
end

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end

function A  = spm_wnorm(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A   = A + 1e-16;
A   = bsxfun(@minus,1./sum(A,1),1./A);
end

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end
end

function [L] = spm_backwards(O,Q,A,B,u,t,T)
% Backwards smoothing to evaluate posterior over initial states
%--------------------------------------------------------------------------
% inverse transition

top = T;
bot = t;
for timestep = top:-1:bot+1
    B_t = B(:,:,u(timestep-1)); 
    Q = Q(:);
    prior = Q'*B_t;
    post = prior.*A{timestep-1}';
    L{timestep-1} = post/sum(post);
    Q = L{timestep-1}';
end
    

% marginal distribution over states
%--------------------------------------------------------------------------
end

% NOTES:

% generate least surprising outcome
%==========================================================================

% or least surprising outcome
%------------------------------------------------------
% j     = sub2ind(Ns(m,:),ind{:});
% F     = zeros(No(m,g),1);
% for i = 1:No(m,g)
%     po    = MDP(m).A{g}(i,:);
%     po    = spm_norm(spm_vec(po));
%     F(i)  = spm_log(po(j));
% end
% po        = spm_softmax(F*512);

