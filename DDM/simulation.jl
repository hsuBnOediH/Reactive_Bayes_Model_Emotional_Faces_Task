#=
1. Task overview
On each of 200 trials, a person hears either a “low” or “high” tone, then sees a face whose expression varies from clearly sad 
to clearly angry (with two levels of ambiguity in between). Their job is to decide, as quickly as possible, whether the face is 
sad or angry. Base on the response, the person will receive a reward after each trail, the reward is higher as the respone time is 
shorter under correct response. If the respone is not correct or the person does not respond, the reward is minimal.
There will be a association between the tone and the face expression, the person will learn this association during the trials.
However, this association is not fixed, it will change during the trials, and the person won't know when the association changes.

2. How the model “learns” the tone–face mapping
•	The model starts with a neutral sense of which tone goes with which emotion.
•	After each trial, a reward is given based on whether the choice was correct and response time.
•	If the choice was correct, the model becomes more confident in that tone–emotion pairing; if not, it becomes less confident.
•	A small “forgetting” factor ensures that old outcomes gradually lose influence, so the model can adapt if the mapping changes 
over time.

3. How the model “decides” on each trial
•	in the RL model, it will maitain two beliefs, one for each tone–emotion pairing.(associations), each association have two possible case
    one for the high-angry, low-sad association, and one for the high-sad, low-angry association.
•	when the tone is given, each association will collapse to one of the two possible cases, depending on the tone.
•	The rl will effect the attribute of the DDM model, such as the drift rate and boundary separation.
•	The model samples from a drift-diffusion model (DDM) to simulate the decision-making process.
•	The generated response is either “angry face” or “sad face,” along with a response time will then be use to feedback the reward by compare with the observed data(ground truth).

=#

using Random                              
using RxInfer        
using SequentialSamplingModels   
using StatsBase: countmap
using Plots

# fix random seed for reproducibility
rng = MersenneTwister(1234)              

#=
Example of Data
+--------------+----------+------+---------+
| Trial Number | Observed | Tone |Intensity|
+--------------+----------+------+---------+
|      1       |    1     |  1   |   0.75  |
|      2       |    2     |  2   |   0.25  |
+--------------+----------+------+---------+

Trial Number: the index of the trial, from 1 to 200, the sequence matters

INPUT:
----------
Observed:
    1: the Ground Truth of current trail is in "high-angry" or "low-sad" association
    2: the Ground Truth of current trail is in "high-sad", "low-angry" association

Tone:
    1: in this trail, the subject observed a low tone
    2: in this trail, the subject observed a high tone

Intensity:
    1: the face is clealy a high-sad, low-angry setting face
    0.75: the face is ambiguous a high-sad, low-angry setting face
    0.25:the face is ambiguous a high-angry, low-sadsetting face
    0:  the face is clealy a high-angry, low-sadsetting face

OUTPUT:
----------
Response:
    1: the subject response "angry face" in the current trial
    2: the subject response "sad face" in the current trial
    Nan: the subject not respond in the current trial
Response Time:
    time in second of the whole action
=# 
n_trials       = 200 
intensity_data = rand(rng, [0.0, 0.25, 0.75, 1.0], n_trials) 
observed_data  = rand(rng, Bernoulli(0.5), n_trials) 
tone_data     = rand(rng, Bernoulli(0.5), n_trials)  # tone data, not used in this simulation 

#=
RL-DDM model struct

RL Model:
    - eta: learning rate
    - inv_temp: inverse temperature
    - V0: initial belief
    - omega: forgetting rate
DDM Model:
    - T: non-decision time
    - w: starting bias
    - a: boundary separation
    
Connection between RL and DDM:
    - drift_rate_scaler_low: drift  rate scaler when tone is low
    - drift_rate_scaler_high: drift rate scaler when tone is high
    - boundary_scaler_low: boundary scaler when tone is low
    - boundary_scaler_high: boundary scaler when tone is high
=# 

# RL parameters
eta  = 0.2   
inv_temp  = 2.0   
V0 = 0.5    
omega  = 0.1    
# DDM parameters 
T = 0.3   
w   = 0.5  
v = 0.2 
a = 1.0 

# Connection parameters

drift_rate_scaler_low  = -0.5 
drift_rate_scaler_high = 0.5  
boundary_scaler_low    = 0.3 
boundary_scaler_high  = -0.2
# zero-initialized vectors for simulation results
choice_sim = Vector{Int}(undef, n_trials)
rt_sim     = Vector{Float64}(undef, n_trials)
choice_association = Vector{Int}(undef, n_trials)

# q is two value vector, each respresenting the belief for each choice, initialized to V0
# where q[1] is for high-angry, low-sad association, q[2] is for high-sad, low-angry association   
q = fill(V0, 2)  # belief vector for two choices (0 and 1)
# record belief and accuracy histories
q1_hist = zeros(Float64, n_trials)
q2_hist = zeros(Float64, n_trials)
accuracy = falses(n_trials)
for t in 1:n_trials
    # softmax the q value 
    association_prob = exp.(inv_temp .* q) ./ sum(exp.(inv_temp .* q))
    # given the tone, either high or low, two of the association choices will collpase to one
    tone_at_t = tone_data[t]  
    if tone_at_t == 1  # low tone
        sad_face_prob = association_prob[1]  # high-angry, low-sad association
        angry_face_prob = association_prob[2]  # high-sad, low-angry association
    else  # high tone
        sad_face_prob = association_prob[2]  # high-sad, low-angry association
        angry_face_prob = association_prob[1]  # high-angry, low-sad association
    end

    # 1) Compute trial-specific drift rate base on the tone and intensity
    k = tone_at_t == 1 ? drift_rate_scaler_low : drift_rate_scaler_high
    v_t = v + k * intensity_data[t]            

    # 2) Compute belief-modulated boundary
    choice_prob_diff = sad_face_prob - angry_face_prob  # difference in choice probabilities
    choice_prob_diff = clamp(choice_prob_diff, -1.0, 1.0)  # clamp to avoid extreme values                 
    if tone_at_t == 1  # low tone
        bs = boundary_scaler_low * choice_prob_diff  # boundary modulator for high-angry, low-sad
    else  # high tone
        bs = boundary_scaler_high * choice_prob_diff  # boundary modulator for high-sad, low-angry
    end
    a_t = a * (1 + bs)  # boundary separation modulated by belief difference        

    # 3) Sample from the DDM via SSM.jl
    dist = DDM(v_t, a_t, w, T)              
    rnd  = rand(rng, dist)                  
    choice_sim[t], rt_sim[t] = rnd.choice, rnd.rt

    # 4) reward computation
    # give the choice is 1 or 2 combine with tone, compute which association the subject is responding to
    if tone_at_t == 1  # low tone
        choice_association[t] = (choice_sim[t] == 1) ? 1 : 2  # high-angry, low-sad association
    else  # high tone
        choice_association[t] = (choice_sim[t] == 1) ? 2 : 1  # high-sad, low-angry association
    end

    # TODO: Reward computation add time factor?
    # compare the choice_association with the observed data to determine reward
    reward = observed_data[t] == choice_association[t] ? 1.0 : 0.0 
    # record whether this trial was correct and the pre-update beliefs
    accuracy[t] = (reward == 1.0)
    q1_hist[t] = q[1]
    q2_hist[t] = q[2]
    println("Trial $t: Choice = $(choice_sim[t]), RT = $(rt_sim[t]), Reward = $reward, Association = $(choice_association[t])")
    # 5) Belief update (Rescorla–Wagner + forgetting)             
    # if win, update belief towards the choice, else update away from the choice
    q = q .+ eta * (reward .- q)  # update belief based on reward
    q = (1 - omega) .* q .+ omega .* V0  # apply forgetting
    # ensure q stays within bounds
    q = clamp.(q, 0.0, 1.0)  # ensure belief stays within [0, 1]


    # if lose update belief away from the choice
    q = q .- eta * (reward .- q)  # update belief based on reward
    q = (1 - omega) .* q .+ omega .* V0  # apply forgetting
    # ensure q stays within bounds
    q = clamp.(q, 0.0, 1.0)  # ensure belief stays within [0, 1]
end


##### Visualization of Simulation Results #####
using Plots

# 1) Response time distribution
histogram(rt_sim;
    bins=30,
    xlabel="Response time (seconds)",
    ylabel="Count",
    title="Distribution of Response Times",
    legend=false)
savefig("rt_distribution.png")

# 2) Choice (association) frequencies
counts = countmap(choice_association)
bar(
    collect(keys(counts)),
    collect(values(counts));
    xlabel="Chosen Association (1=high-angry/low-sad, 2=high-sad/low-angry)",
    ylabel="Frequency",
    title="Choice Frequencies",
    xticks=[1,2]
)
savefig("choice_frequencies.png")

# 3) Learning curve (cumulative accuracy over trials)
learning_curve = cumsum(accuracy) ./ (1:n_trials)
plot(
    1:n_trials,
    learning_curve;
    xlabel="Trial Number",
    ylabel="Cumulative Accuracy",
    title="Learning Curve Over Trials",
    legend=false
)
savefig("learning_curve.png")

# 4) Belief trajectories over trials
plt1 = plot(
    1:n_trials,
    q1_hist;
    label="Belief: high-angry/low-sad",
    xlabel="Trial Number",
    ylabel="Belief Value",
    title="Belief Trajectories"
)
plot!(
    1:n_trials,
    q2_hist;
    label="Belief: high-sad/low-angry"
)
savefig("belief_trajectories.png")

println("Saved plots: rt_distribution.png, choice_frequencies.png, learning_curve.png, belief_trajectories.png")
