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
using Plots
using Distributions
using CSV, DataFrames

# fix random seed for reproducibility
rng = MersenneTwister(1234)              
max_allowed_rt =0.8
missing_response_reward = -2.00  # reward for missing response, can be set to 0 or a small negative value
incorrect_response_reward = -0.50  # fixed reward for incorrect response, can be set to a negative value
max_correct_response_reward = 0.75  # maximum reward for correct response, can be set to a positive value
min_correct_response_reward = -0.25  # minimum reward for correct response, can be set to a negative value

#=
Example of Data
+--------------+----------+------+---------+
| Trial Number | Observed | Tone |Intensity|
+--------------+----------+------+---------+
|      1       |    1     |  1   |   0.25  |
|      2       |    2     |  2   |   0.75  |
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
    -0.75:the face is ambiguous a high-angry, low-sads etting face
    -1:  the face is clealy a high-angry, low-sad setting face

OUTPUT:
----------
Response:
    1: the subject response "angry face" in the current trial
    2: the subject response "sad face" in the current trial
    Nan: the subject not respond in the current trial
Response Time:
    time in second of the whole action
=# 
trial_file = "emotional_faces_processed_data/task_data_53b98f20fdf99b472f4700e4_responses.csv"          
df = CSV.read(trial_file, DataFrame)
observed_data = Int.(df.observed) .+ 1 
n_trials = 200  # number of trials
intensity_data = Vector{Float16}(undef, n_trials)
for i in 1:n_trials
    if observed_data[i] == 1  # high-angry, low-sad association
        # pick one of the two discrete intensities -1.0 or -0.75
        intensity_data[i] = Float16(rand(rng, (-1.0, -0.75)))
    else  # high-sad, low-angry association
        # pick one of the two discrete intensities 0.75 or 1.0
        intensity_data[i] = Float16(rand(rng, (0.75, 1.0)))
    end
end
tone_data     = rand(rng, Bernoulli(0.5), n_trials)  # tone data, not used in this simulation 

#=
RL-DDM model struct
RL Model:
    - eta_win: learning rate for winning trials
    - eta_loss: learning rate for losing trials
    - inv_temp: inverse temperature
    - V0: initial belief
    - omega: forgetting factor
DDM Model:
    - T: non-decision time
    - w: starting bias
    - a: base boundary separation
    - v: base drift rate
Connection between RL and DDM:
    - drift_rate_scaler_low: drift  rate scaler when tone is low
    - drift_rate_scaler_high: drift rate scaler when tone is high
    - boundary_scale：r: boundary separation scaler based on the belief difference
=# 

# RL parameters
eta_win = 0.05  # learning rate for winning trials
eta_loss = 0.2  # learning rate for losing trials
inv_temp  = 2.0   
V0 = 0.5    
omega  = 0.1    
# DDM parameters 
T = 0.3 
w_base   = 0.5  
v_base   = 0.2 
a_base   = 1.0 

# Connection parameters
drift_rate_scaler_low  = -0.5 
drift_rate_scaler_high = 0.5  
boundary_scaler    = 0.3
bias_scaler = 0.2

# zero-initialized vectors for simulation results
choice_sim = Vector{Int}(undef, n_trials)
rt_sim     = Vector{Float64}(undef, n_trials)
choice_association = Vector{Int}(undef, n_trials)

# q is two value vector, each respresenting the belief for each choice, initialized to V0
# where q[1] is for high-angry, low-sad association, 
# q[2] is for high-sad, low-angry association   
q = fill(V0, 2)
# record belief and accuracy histories
q1_hist = zeros(Float64, n_trials)
q2_hist = zeros(Float64, n_trials)
accuracy = falses(n_trials)
# record additional histories for visualization
sad_face_prob_hist = zeros(Float64, n_trials)
angry_face_prob_hist = zeros(Float64, n_trials)
v_t_hist = zeros(Float64, n_trials)
a_t_hist = zeros(Float64, n_trials)
association_1_prob_hist = zeros(Float64, n_trials)
association_2_prob_hist = zeros(Float64, n_trials)

for t in 1:n_trials
    # println("q at trial $t: $(q)")
    # softmax the q value 
    z = inv_temp .* q
    z .-= maximum(z)                          # shift so max(z)==0
    exp_z = exp.(z)                           # now no overflow
    association_prob = exp_z ./ sum(exp_z)
    association_1_prob_hist[t] = association_prob[1]  # high-angry, low-sad association
    association_2_prob_hist[t] = association_prob[2]  # high-sad, low-angry association
    # println("association_prob: $association_prob")
    # given the tone, either high or low, two of the association choices will collpase to one
    tone_at_t = tone_data[t]  
    if tone_at_t == 1  # low tone
        # [1]: high-angry, low-sad association --> low_sad
        # [2]: high-sad, low-angry association --> low_angry
        sad_face_prob = association_prob[1]  
        angry_face_prob = association_prob[2]  
    else  # high tone
        # [1]: high-angry, low-sad association --> high_angry
        # [2]: high-sad, low-angry association --> high_sad
        sad_face_prob = association_prob[2]  
        angry_face_prob = association_prob[1]
    end
    # println("sad_face_prob: $sad_face_prob, angry_face_prob: $angry_face_prob")
    # record collapse probabilities
    sad_face_prob_hist[t] = sad_face_prob
    angry_face_prob_hist[t] = angry_face_prob
    
    k = tone_at_t == 1 ? drift_rate_scaler_low : drift_rate_scaler_high
    v_t = v_base + k * intensity_data[t]            

    # 2) Compute belief-modulated boundary
    # use the Q values difference to determine the boundary separation, lower bounary - upper boundary
    choice_prob_diff =  angry_face_prob -sad_face_prob
    # println("a before modulation: $a, choice_prob_diff: $choice_prob_diff, tone_at_t: $tone_at_t, boundary_scaler: $boundary_scaler")
    a_t = a_base + boundary_scaler * choice_prob_diff  # boundary separation modulated by belief difference

    # record trial-specific DDM parameters
    v_t_hist[t] = v_t
    a_t_hist[t] = a_t
    
    # check if the them add up to 1, if not, normalize them
    if sad_face_prob + angry_face_prob != 1.0
        println("add up is not 1, normalizing sad_face_prob and angry_face_prob, sad_face_prob: $sad_face_prob, angry_face_prob: $angry_face_prob,total: $(sad_face_prob + angry_face_prob)")
        sad_face_prob /= (sad_face_prob + angry_face_prob)
        angry_face_prob /= (sad_face_prob + angry_face_prob)
    end
    # w = 0.0 means bias towards sad face, w = 1.0 means bias towards angry face
    w = w_base + bias_scaler * angry_face_prob 

    # 3) Sample from the DDM via SSM.jl
    # println("Trial $t: Tone = $(tone_at_t), Intensity = $(intensity_data[t]), Drift Rate = $(v_t), Boundary Separation = $(a_t)")
    dist = DDM(v_t, a_t, w, T)          
        
    rnd  = rand(rng, dist)                  
    choice_sim[t], rt_sim[t] = rnd.choice, rnd.rt
    # println("Trial $t: Tone = $(tone_at_t), Intensity = $(intensity_data[t]), Choice = $(choice_sim[t]), RT = $(rt_sim[t])")

    # 4) reward computation
    # check if the sample response time exceeds the maximum allowed response time
    # if it does, mark the choice_association as 0 (no association chosen), this trail will be considered as a missing response
    # otherwise, determine the association based on the tone and choice, determine the correctness of the choice, and compute the reward
    # the reward is based on the response time and correctness of the choice
    if rt_sim[t] > max_allowed_rt
        choice_association[t] = 0  # no association chosen, mark as 0
        accuracy[t] = false  # if response time exceeds the maximum allowed, mark as incorrect
        reward = missing_response_reward
        # println("Trial $t: Response time exceeds maximum allowed, marking as no association chosen.")
    else
        if tone_at_t == 1  # low tone
            choice_association[t] = (choice_sim[t] == 1) ? 1 : 2  # high-angry, low-sad association
        else  # high tone
            choice_association[t] = (choice_sim[t] == 1) ? 2 : 1  # high-sad, low-angry association
        end
        correctness =  observed_data[t] == choice_association[t] ? 1.0 : 0.0 
        if correctness == 1.0
            # if the choice is correct, reward is based on the response time
            accuracy[t] = true
            reward = ((( min_correct_response_reward- max_correct_response_reward) / (max_allowed_rt - 0)) * rt_sim[t])+ max_correct_response_reward
        else
            # if the choice is incorrect, use the fixed reward for incorrect response
            accuracy[t] = false
            reward = incorrect_response_reward
        end
    end
   
    q1_hist[t] = q[1]
    q2_hist[t] = q[2]
    println("Trial $t: Choice = $(choice_sim[t]), RT = $(rt_sim[t]), Reward = $reward, Association = $(choice_association[t])")

    # 5) Belief update (Rescorla–Wagner + forgetting)             
    # update the beliefs based on the association chosen --> which q to update
    # based on if winning or losing ---> which learning rate to use
    # for the unchosen association, forget to V0 for both associations q values
    if choice_association[t] == 1  # high-angry, low-sad association
        # println("chosen association: high-angry, low-sad association, 1")
        if observed_data[t] == 1  # high-angry, low-sad association
            # println("observed data: high-angry, low-sad association, correct choice")
            # update belief towards the choice high-angry, low-sad association, 1
            q[1] = q[1] + eta_win * (reward - q[1])  # update belief based on reward
        else
            # println("observed data: high-sad, low-angry association, incorrect choice")
            # update belief away from the choice high-angry, low-sad association, 1
            q[1] = q[1] + eta_loss * (reward - q[1])  # update belief based on reward
        end
    elseif choice_association[t] == 2  # high-sad, low-angry association
        # println("chosen association: high-sad, low-angry association, 2")
        if observed_data[t] == 2  # high-sad, low-angry association
            # println("observed data: high-sad, low-angry association, correct choice")
            q[2] = q[2] + eta_win * (reward - q[2]) 
        else
            # println("observed data: high-angry, low-sad association, incorrect choice")
            # update belief away from the choice high-sad, low-angry association, 2
            q[2] = q[2] + eta_loss * (reward - q[2])
        end
    else
        # println("No association chosen, no update")
        # for unchose association, forget to V0
        q[1] = (1 - omega) * q[1] + omega * V0  # forgetting for high-angry, low-sad association
        q[2] = (1 - omega) * q[2] + omega * V0  # forgetting for high-sad, low-angry association
    end
    println("Updated beliefs: q1 = $(q[1]), q2 = $(q[2])")
    println("--------------------------------------------------")
end

trials = 1:n_trials

# 1. Q-value histories
# p1 = plot(trials, q1_hist, label="q₁ (high-angry)", xlabel="Trial", ylabel="Q value", title="Belief Histories")
# plot!(p1, trials, q2_hist, label="q₂ (high-sad)")

# 2. Collapse probabilities
# p2 = plot(trials, sad_face_prob_hist, label="P(sad)", xlabel="Trial", ylabel="Probability", title="Collapsed Probabilities")
# plot!(p2, trials, angry_face_prob_hist, label="P(angry)")

# 3. DDM parameters over trials
# p3 = plot(trials, v_t_hist, label="Drift rate vₜ", xlabel="Trial", ylabel="Value", title="Drift Rate Over Trials")
# p4 = plot(trials, a_t_hist, label="Boundary aₜ", xlabel="Trial", ylabel="Value", title="Boundary Separation Over Trials")

rt1 = rt_sim[choice_sim .== 1]
rt2 = rt_sim[choice_sim .== 2]

bins = minimum(rt_sim):0.05:maximum(rt_sim)
p_hist = histogram(
  rt1; bins=bins, label="Choice 1",
  xlabel="RT (s)", ylabel="Count",
  title="RT Distribution by Choice", opacity=0.6,
)
histogram!(
  p_hist, rt2; bins=bins, label="Choice 2", opacity=0.6,
)
vline!(
  p_hist, [0.8]; label="Threshold 0.8 s",
  linestyle=:dash, linewidth=2, color=:red,
)

using StatsBase
true_face = Vector{String}(undef, n_trials)
for i in 1:n
    if observed_data[i] == 1
        # assoc “high-angry / low-sad”
        true_face[i] = tone_data[i] == 2 ? "angry" : "sad"
    else
        # assoc “high-sad / low-angry”
        true_face[i] = tone_data[i] == 2 ? "sad"   : "angry"
    end
end

ground_truth_face_counts = countmap(true_face .== "angry")
simulated_face_counts    = countmap(choice_sim .== 1)

# extract the “angry” counts
gt_angry = ground_truth_face_counts[ true ]
gt_sad  = ground_truth_face_counts[ false ]
sim_angry = simulated_face_counts[ true ]
sim_sad  = simulated_face_counts[ false ]
counts = [
    gt_angry  gt_sad;
    sim_angry sim_sad
]

# now plot 2 groups × 2 series → 4 bars total
p_bar = bar(
    ["Ground Truth", "Simulated"],  # two x-categories
    counts,                         # 2×2 matrix of heights
    label     = ["Angry", "Sad"],   # one label per column
    xlabel    = "Source",
    ylabel    = "Count",
    title     = "Face Type Counts",
    legend    = :topright,
    bar_width = 0.6,
)

# ─── 5. stack histogram above bar chart ───────────────────────────────────────
plot(
  p_hist, p_bar;
  layout = @layout([a{0.7h}; b{0.3h}]),
  size   = (700, 600),
)







zeros_idx    = findall(choice_association .== 0)
nonzeros_idx = findall(choice_association .!= 0)
# 5. Observed vs. Simulated Choices
# base plot
p6 = scatter(
    trials,
    observed_data;
    size = (1200, 800),
    label="Observed",
    marker=:circle,
    markersize=4,
    markerstrokewidth=0,         # no outline
    xlabel="Trial",
    ylabel="Association (1=high-angry/low-sad, 2=high-sad/low-angry, 0=no response)",
    title="Choices Comparison"
)

# for choice_association[nonzeros_idx], if the value is 1, mins 0.05, if 2, add 0.05
scaled_choice_association = zeros(Float64, n_trials)
for i in nonzeros_idx
    if choice_association[i] == 1
        scaled_choice_association[i] = 1 - 0.05  # high-angry, low-sad association
    elseif choice_association[i] == 2
        scaled_choice_association[i] = 2 + 0.05  # high-sad, low-angry association
    end 
end
scatter!(
    p6,
    trials[nonzeros_idx],
    scaled_choice_association[nonzeros_idx];
    label="Simulated",
    marker=:dot,
    markersize=4,
    markerstrokewidth=0,
    alpha=0.7,                   # more transparent
    color=:red
)

# no-response zeros in translucent green dots
scatter!(
    p6,
    trials[zeros_idx],
    choice_association[zeros_idx];
    label="No response",
    marker=:dot,
    markersize=3,
    markerstrokewidth=0,
    alpha=0.3,
    color=:green
)

# intensity (shifted) as small black crosses
scaled_intensity = intensity_data .+ 1.5
scatter!(
    p6,
    trials,
    scaled_intensity;
    label="Intensity (shifted)",
    marker=:xcross,
    color=:orange,        # brighter than black
    markersize=6,         # bigger so it’s easier to spot
    alpha=0.8,            # more opaque
    markerstrokewidth=1   # ensure the stroke is visible
)

# addjust the prob_association_1 and prob_association_2 
scaled_association_1_prob_hist = association_1_prob_hist .+ 1
scaled_association_2_prob_hist = association_2_prob_hist .+ 1
plot!(
    trials,
    scaled_association_1_prob_hist;
    label="prob_association_1",
    linewidth=3,
    linestyle=:dash,
    color=RGBA(1, 0, 0, 0.9)   # almost opaque red
)

plot!(
    trials,
    scaled_association_2_prob_hist;
    label="prob_association_2",
    linewidth=3,
    linestyle=:dash,
    color=RGBA(1, 1, 0, 0.9)   # almost opaque yellow
)



# optionally move legend out of the way
plot!(p6,
    legend = :outerbottom,
    legendcols = 1         # if you want each entry on its own line; bump up if you’d prefer side-by-side
)









# 6. Intensity and Tone
# p8 = scatter(trials, tone_data, label="Tone (0=low,1=high)", xlabel="Trial", ylabel="Tone", title="Tone Over Trials")

# display all plots
# display(p1)
# display(p2)
# display(p3)
# display(p4)
# display(p5)
# display(p6)
# display(p7)
# display(p8)
