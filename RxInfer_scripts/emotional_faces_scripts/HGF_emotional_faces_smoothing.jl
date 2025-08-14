# import Pkg
# Pkg.add(["RxInfer", "DataFrames", "CSV", "Plots", "StatsPlots", "Distributions", 
# "BenchmarkTools", "StableRNGs", "ExponentialFamilyProjection", "ExponentialFamily",
# "ReactiveMP", "Cairo", "GraphPlot", "Random"])
# Activate local environment, see `Project.toml`
import Pkg; 

# determine if running on cluster or locally
if get(ENV, "CLUSTER", "false") == "true"
    println("Running on the cluster...")
    # println(ENV)
    println("SUBJECT: ", get(ENV, "SUBJECT", "NOT SET"))
    println("PREDICTIONS_OR_RESPONSES: ", get(ENV, "PREDICTIONS_OR_RESPONSES", "NOT SET"))
    println("RESULTS: ", get(ENV, "RESULTS", "NOT SET"))

    Pkg.activate("/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/cluster_environment/")
    subject_id = get(ENV, "SUBJECT", "SUBJECT_NOT_SET")
    if length(subject_id) == 5
        study = "local"
    elseif length(subject_id) == 24
        study = "prolific"
    else
        error("Invalid subject ID length. Must be 5 (local) or 24 (prolific).")
    end
    predictions_or_responses = get(ENV, "PREDICTIONS_OR_RESPONSES","PREDICTIONS_OR_RESPONSES_NOT_SET")
    results_dir = get(ENV, "RESULTS","RESULTS_NOT_SET")
    root = "/media/labs/"
    formatted_time = ""
    hyperparam_str = get(ENV, "HYPERPARAM_STR", "NOT SET")
    println("Hyperparams String: ", hyperparam_str)
    hyperparam_dict = Dict(
        pair[1] => parse(Float64, pair[2]) for pair in 
        (split(kv, "=") for kv in split(hyperparam_str, ","))
    )
    # Assign prior parameter values and initialize the marginal posteriors for some parameters
    prior_kappa_mean = get(hyperparam_dict, "prior_kappa_mean", "NOT SET")
    prior_kappa_variance = get(hyperparam_dict, "prior_kappa_variance", "NOT SET")
    prior_omega_mean = get(hyperparam_dict, "prior_omega_mean", "NOT SET")
    prior_omega_variance = get(hyperparam_dict, "prior_omega_variance", "NOT SET")
    prior_beta_shape = get(hyperparam_dict, "prior_beta_shape", "NOT SET")
    prior_beta_rate = get(hyperparam_dict, "prior_beta_rate", "NOT SET")
    prior_z_initial_mean = get(hyperparam_dict, "prior_z_initial_mean", "NOT SET")
    prior_z_initial_variance = get(hyperparam_dict, "prior_z_initial_variance", "NOT SET")
    prior_x_initial_mean = get(hyperparam_dict, "prior_x_initial_mean", "NOT SET")
    prior_x_initial_variance = get(hyperparam_dict, "prior_x_initial_variance", "NOT SET")
    initialized_z_mean = get(hyperparam_dict, "initialized_z_mean", "NOT SET")
    initialized_z_precision = get(hyperparam_dict, "initialized_z_precision", "NOT SET")
    initialized_kappa_mean = get(hyperparam_dict, "initialized_kappa_mean", "NOT SET")
    initialized_kappa_variance = get(hyperparam_dict, "initialized_kappa_variance", "NOT SET")
    initialized_omega_mean = get(hyperparam_dict, "initialized_omega_mean", "NOT SET")
    initialized_omega_variance = get(hyperparam_dict, "initialized_omega_variance", "NOT SET")
    initialized_beta_shape = get(hyperparam_dict, "initialized_beta_shape", "NOT SET")
    initialized_beta_rate = get(hyperparam_dict, "initialized_beta_rate", "NOT SET")
    niterations = get(hyperparam_dict, "niterations", "NOT SET")
    niterations = Int(niterations) # cast as an integer
else
    println("Running locally...")
    Pkg.activate("L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/"); # Note that the Project and Manifest files are in the same directory as this script.
    # Click Julia: Activate this Environment to run the REPL
    subject_id = "AA003" # Use a prolific or local subject id e.g., 5a5ec79cacc75b00017aa095
    # Local subject IDs will always be 5 characters, while prolific IDs will always be 24
    if length(subject_id) == 5
        study = "local"
    elseif length(subject_id) == 24
        study = "prolific"
    else
        error("Invalid subject ID length. Must be 5 (local) or 24 (prolific).")
    end

    predictions_or_responses = "responses" # Haven't set up infrastructure to fit predictions
    results_dir = "L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_test"
    root = "L:/"
    # get current datetime
    using Dates
    now_time = now()
    formatted_time = "_" * Dates.format(now_time, "yyyy-mm-ddTHH_MM_SS")
    # Assign prior parameter values and initialize the marginal posteriors for some parameters
    prior_z_precision_shape = 1.0
    prior_z_precision_rate = 1.0
    initialized_z_precision_shape = 1.0
    initialized_z_precision_rate = 1.0
    prior_kappa_mean = 1.0
    prior_kappa_variance = 0.1
    prior_omega_mean = 0.0
    prior_omega_variance = .01 # used to be .01
    prior_beta_shape = 1.0
    prior_beta_rate = 1.0
    prior_z_initial_mean = 0.0
    prior_z_initial_variance = 1.0
    prior_x_initial_mean = 0.0
    prior_x_initial_variance = 1.0
    initialized_z_mean = 0.0
    initialized_z_precision = 1.0
    initialized_kappa_mean = 1.0
    initialized_kappa_variance = 0.1
    initialized_omega_mean = 0.0
    initialized_omega_variance = 0.01
    initialized_beta_shape = 0.1
    initialized_beta_rate = 0.1
    niterations = 3
end
Pkg.instantiate()  # Reinstall missing dependencies


using RxInfer
using ExponentialFamilyProjection, ExponentialFamily
using Distributions
using Plots, StatsPlots
using StableRNGs
using CSV, DataFrames

# Import file containing callbacks functions and tagged logger
include(root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/callbacks.jl")

# Seed for reproducibility
seed = 23
rng = StableRNG(seed)

# Read in the task data
if study == "local"
    file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_local/local_emotional_faces_processed_data_06-11-25-18_21_28/task_data_$(subject_id)_processed_data.csv"
elseif study == "prolific"
    file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/prolific_emotional_faces_processed_data_06-11-25-18_23_16/task_data_$(subject_id)_processed_data.csv"
end

data = CSV.read(file_name, DataFrame)
# Create variable for observations which is equal to 1 (congruent, high intensity), .75 (congruent, low intensity), .25 (incongruent, low intensity), 0 (incongruent, high intensity)
#obs_data = data.observed 
obs_data = data.face_intensity
obs_data = Float64.(obs_data)
# Create variable for responses
resp_data = data.resp_sad_high_or_angry_low
resp_data = Float64.(resp_data)

# For debugging, let's just use two observations and responses
# resp_data = resp_data[1:2]
# obs_data = obs_data[1:2]

# Count the number of missing responses
nan_responses = count(isnan, resp_data)
println("Number of NaN Responses: ", nan_responses)

# Exit if participant has 200 mising responses
if nan_responses == 200
    println("Too many missing responses. Exiting.")
    exit()
end


# Replace NaN responses with the missing keyword
resp_data = convert(Vector{Union{Missing, Float64}}, resp_data)
resp_data .= ifelse.(isnan.(resp_data), missing, resp_data)

# If there are missing responses, we won't be able to compute Bethe Free Energy values because the graph is incomplete
# The RxInfer developers seem to be working on this issue, so we can check later if it's been resolved
if nan_responses > 0
    compute_bethe_free_energy = false
else
    compute_bethe_free_energy = true
end




@model function hgf_smoothing(obs, resp)
    # Initial states - adjust means to be closer to data range
    z_precision ~ Gamma(shape = prior_z_precision_shape, rate = prior_z_precision_rate) where { pipeline = TaggedLogger("z_precision") }# Reduced variance
    z_initial ~ Normal(mean = prior_z_initial_mean, variance = prior_z_initial_variance)  where { pipeline = TaggedLogger("z_initial") }# Reduced variance
    x_initial ~ Normal(mean = prior_x_initial_mean, variance = prior_x_initial_variance)  where { pipeline = TaggedLogger("x_initial") }# Reduced variance
    
    x_prev = x_initial 
    z_prev = z_initial 

    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = prior_kappa_mean, variance = prior_kappa_variance)  where { pipeline = TaggedLogger("κ") }# Reduced variance and mean
    ω ~ Normal(mean = prior_omega_mean, variance = prior_omega_variance)  where { pipeline = TaggedLogger("ω") }# Reduced variance
    β ~ Gamma(shape = prior_beta_shape, rate = prior_beta_rate)  where { pipeline = TaggedLogger("β") } # Less constrained

    for i in eachindex(obs)
        # Third layer
        z[i] ~ Normal(mean = z_prev, precision = z_precision)  where { pipeline = TaggedLogger("z[$(i)]") }
 
        # Second layer
        x[i] ~ GCV(x_prev, z[i], κ, ω) where { pipeline = TaggedLogger("x[$(i)]") }

        # First layer
        obs[i] ~ Probit(x[i]) where { pipeline = TaggedLogger("obs[$(i)]") }
        temp[i] ~ softdot(β, x[i], 1.0) where { pipeline = TaggedLogger("temp[$(i)]") }
        resp[i] ~ Probit(temp[i]) where { pipeline = TaggedLogger("resp[$(i)]") }

        # Update previous states
        z_prev = z[i]
        x_prev = x[i]
    end 
 end

 # Visualize model
 model_generator = hgf_smoothing() | (obs = [1], resp = [1],)
 model_to_plot   = RxInfer.getmodel(RxInfer.create_model(model_generator))
 using GraphViz
 # Call `load` function from `GraphViz` to visualise the structure of the graph
 GraphViz.load(model_to_plot, strategy = :simple)
 
@constraints function hgfconstraints_smoothing() 
    #Structured mean-field factorization constraints
    q(x, z, temp, κ, ω, β, x_initial,z_precision,z_initial) = q(x, temp, x_initial)q(z_precision)q(z, z_initial)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
end

import ReactiveMP: as_companion_matrix, ar_transition, getvform, getorder, add_transition!
# Add this method to prevent compiler ambiguity for the product of colliding messages
Base.prod(::GenericProd, left::ProductOf{L, R}, ::Missing) where {L, R} = left


@meta function hgfmeta_smoothing()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV() -> GCVMetadata(GaussHermiteCubature(31))
end



function run_inference_smoothing(obs, resp, niterations)
    @initialization function hgf_init_smoothing()
        q(z) = NormalMeanPrecision(initialized_z_mean, initialized_z_precision)
        q(κ) = NormalMeanVariance(initialized_kappa_mean, initialized_kappa_variance)
        q(ω) = NormalMeanVariance(initialized_omega_mean, initialized_omega_variance)
        q(β) = GammaShapeRate(initialized_beta_shape, initialized_beta_rate)
        q(z_precision) = GammaShapeRate(initialized_z_precision_shape, initialized_z_precision_rate)
    end

    return infer(
        model = hgf_smoothing(),  
        data = (obs = obs, resp = UnfactorizedData(resp),),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = niterations,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (x = KeepEach(), z = KeepEach(), ω=KeepEach(), κ=KeepEach(), β=KeepEach(), temp=KeepEach(),z_initial=KeepEach(), x_initial=KeepEach(),z_precision=KeepEach(),),
        free_energy = compute_bethe_free_energy,  # Compute Bethe Free Energy when there are no missing values
        free_energy_diagnostics = nothing, # turns off the error for NaN or inf FE
        showprogress = true,
        callbacks      = (
            before_model_creation = before_model_creation,
            after_model_creation = after_model_creation,
            before_inference = before_inference,
            after_inference = after_inference,
            before_iteration = before_iteration,
            after_iteration = after_iteration,
            before_data_update = before_data_update,
            after_data_update = after_data_update,
            on_marginal_update = on_marginal_update
        ),
        addons = (AddonMemory(),), # uncomment this to get a more detailed description of the message computation
    )
end



# For ease of debugging, output to a file instead of the console
if get(ENV, "CLUSTER", "false") == "true"
    infer_result = run_inference_smoothing(obs_data, resp_data, niterations)
else
    # open the log file (it'll be closed automatically at the end of the do-block)
    infer_result =
    open(results_dir*"/inference_smoothing.log", "w") do io
        redirect_stdout(io) do
        redirect_stderr(io) do
            try
            run_inference_smoothing(obs_data, resp_data, niterations)
            catch err
            # print the error message
            println(io, "ERROR during inference: ", err)
            # print the backtrace
            showerror(io, err, catch_backtrace())
            # rethrow; to stop the julia session
            rethrow()
            end
        end
        end
    end
end



# If free energy was computed, find the last iteration where free energy was neither NaN or infinity
if compute_bethe_free_energy
    free_energy_vals = infer_result.free_energy
    niterations_used = findlast(isfinite, free_energy_vals) # checks for not NaN or inf
else
    niterations_used = niterations
end

# Extract posteriors
x_initial_posterior = infer_result.posteriors[:x_initial][niterations_used]
x_initial_posterior = getfield(x_initial_posterior, :data) # added this line because of addons
z_initial_posterior = infer_result.posteriors[:z_initial][niterations_used]
z_initial_posterior = getfield(z_initial_posterior, :data) # added this line because of addons
x_posterior = infer_result.posteriors[:x][niterations_used]
x_posterior = getfield.(x_posterior, :data) # added this line because of addons
z_posterior = infer_result.posteriors[:z][niterations_used]
z_posterior = getfield.(z_posterior, :data) # added this line because of addons
κ_posterior = infer_result.posteriors[:κ][niterations_used]
κ_posterior = getfield(κ_posterior, :data) # added this line because of addons
ω_posterior = infer_result.posteriors[:ω][niterations_used]
ω_posterior = getfield(ω_posterior, :data) # added this line because of addons
β_posterior = infer_result.posteriors[:β][niterations_used]
β_posterior = getfield(β_posterior, :data) # added this line because of addons
z_precision_posterior = infer_result.posteriors[:z_precision][niterations_used]
z_precision_posterior = getfield(z_precision_posterior, :data) # added this line because of addons
temp_posterior = infer_result.posteriors[:temp][niterations_used]
temp_posterior = getfield.(temp_posterior, :data) # added this line because of addons

# Print mean and standard deviation of parameters
println("Parameter Estimates:")
println("κ: mean = $(mean(κ_posterior)), std = $(std(κ_posterior))")
println("ω: mean = $(mean(ω_posterior)), std = $(std(ω_posterior))")
println("β: shape = $(shape(β_posterior)), scale = $(scale(β_posterior))")
println("z precision: shape = $(shape(z_precision_posterior)), scale = $(scale(z_precision_posterior))")
println("x initial: mean = $(mean(x_initial_posterior)), std = $(std(x_initial_posterior))")
println("z initial: mean = $(mean(z_initial_posterior)), std = $(std(z_initial_posterior))")
# Plot the hidden states x and z
p1 = plot(mean.(x_posterior), ribbon=2*std.(x_posterior), 
    label="x (Lower layer)", title="Hidden States", 
    fillalpha=0.3)
plot!(p1, mean.(z_posterior), ribbon=2*std.(z_posterior), 
    label="z (Higher layer)", fillalpha=0.3)
scatter!(p1, 1:length(obs_data), obs_data, label="Observations", 
    markersize=4)
scatter!(p1, 1:length(resp_data), resp_data, label="Responses", 
    markersize=4)

# Plot parameter posteriors
p2 = plot(
    histogram(rand(κ_posterior, 1000), title="κ", label=""),
    histogram(rand(ω_posterior, 1000), title="ω", label=""),
    histogram(rand(β_posterior, 1000), title="β", label=""),
    histogram(rand(z_precision_posterior, 1000), title="z_precision_posterior", label=""),
    histogram(rand(x_initial_posterior, 1000), title="Initial X", label=""),
    histogram(rand(z_initial_posterior, 1000), title="Initial Z", label=""),
    layout=(2,3)
)

plot(p1, p2, layout=(2,1), size=(800,600))
savefig(joinpath(results_dir, "hidden_states_and_parameters_" * subject_id * formatted_time * ".png"))


## Investigate Accuracy of the Model
# Extract free energy values when they were recorded
if compute_bethe_free_energy
    free_energy_vals = infer_result.free_energy
    println("Free Energy values: ", free_energy_vals)

    # Plot Free Energy to check convergence
    plot(free_energy_vals, title="Free Energy over Iterations", xlabel="Iteration", ylabel="Free Energy", label="")
    savefig(joinpath(results_dir, "free_energy_" * subject_id * formatted_time * ".png"))

end

# Get the probability assigned to each response, accounting for missing responses
# Filter out Uninformative() values and NaN responses in a single step
filtered_data = [(t, resp) for (t, resp) in zip(temp_posterior, resp_data) if !(t isa Uninformative) && !ismissing(resp)]
# Extract valid temp_posterior and resp_data values
filtered_temp_posterior, filtered_resp_data = first.(filtered_data), last.(filtered_data)
# Compute probabilities
probit_prob = [cdf(Normal(0,1), t.xi / t.w) for t in filtered_temp_posterior]
# Compute action probabilities
action_probability_values = [resp == 1 ? p : 1 - p for (resp, p) in zip(filtered_resp_data, probit_prob)]
# Mean action probability
mean_action_prob = mean(action_probability_values)
println("Mean action probability: ", mean_action_prob)
# Compute mean model accuracy, the proportion of choices that the model assigned a probability greater than .5
mean_model_accuracy = mean(action_probability_values .> 0.5)

# Get the free energy value if it was computed, otherwise leave NaN
if compute_bethe_free_energy
    bethe_free_energy = [free_energy_vals[niterations_used]]
else
    bethe_free_energy = NaN
end

## save results
results_df = DataFrame(
    id = subject_id,
    number_missing_responses = nan_responses,
    num_iterations = niterations,
    num_iterations_used = niterations_used,
    mean_action_prob = mean(action_probability_values),
    mean_model_accuracy = mean_model_accuracy,
    free_energy = bethe_free_energy,
    kappa_mean = mean(κ_posterior),
    kappa_std = std(κ_posterior),
    omega_mean = mean(ω_posterior),
    omega_std = std(ω_posterior),
    beta_shape = shape(β_posterior), 
    beta_scale = scale(β_posterior),
    x_initial_mean = mean(x_initial_posterior),
    x_initial_std = std(x_initial_posterior),
    z_initial_mean = mean(z_initial_posterior),
    z_initial_std = std(z_initial_posterior),
    prior_kappa_mean = prior_kappa_mean,
    prior_kappa_variance = prior_kappa_variance,
    prior_omega_mean = prior_omega_mean,
    prior_omega_variance = prior_omega_variance,
    prior_beta_shape = prior_beta_shape,
    prior_beta_rate = prior_beta_rate,
    prior_z_initial_mean = prior_z_initial_mean,
    prior_z_initial_variance = prior_z_initial_variance,
    prior_x_initial_mean = prior_x_initial_mean,
    prior_x_initial_variance = prior_x_initial_variance,
    initialized_z_mean = initialized_z_mean,
    initialized_z_precision = initialized_z_precision,
    initialized_kappa_mean = initialized_kappa_mean,
    initialized_kappa_variance = initialized_kappa_variance,
    initialized_omega_mean = initialized_omega_mean,
    initialized_omega_variance = initialized_omega_variance,
    initialized_beta_shape = initialized_beta_shape,
    initialized_beta_rate = initialized_beta_rate,
    niterations = niterations,
)



# Save the results as a CSV 
output_file = joinpath(results_dir, "model_results_" * subject_id * formatted_time * ".csv")
CSV.write(output_file, results_df)

# Call the filtering function using the posterior parameter values
include(root*"rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/HGF_emotional_faces_filtering.jl")



