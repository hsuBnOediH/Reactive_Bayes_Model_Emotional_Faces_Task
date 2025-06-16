# import Pkg
# Pkg.add(["RxInfer", "DataFrames", "CSV", "Plots", "StatsPlots", "Distributions", 
# "BenchmarkTools", "StableRNGs", "ExponentialFamilyProjection", "ExponentialFamily",
# "ReactiveMP", "Cairo", "GraphPlot", "Random"])
# Activate local environment, see `Project.toml`
import Pkg; 
Pkg.build("MATLAB") 

# determine if running on cluster or locally
if get(ENV, "CLUSTER", "false") == "true"
    println("Running on the cluster...")
    # println(ENV)
    println("SUBJECT: ", get(ENV, "SUBJECT", "NOT SET"))
    println("PREDICTIONS_OR_RESPONSES: ", get(ENV, "PREDICTIONS_OR_RESPONSES", "NOT SET"))
    println("RESULTS: ", get(ENV, "RESULTS", "NOT SET"))

    Pkg.activate("/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/cluster_environment/")
    subject_id = get(ENV, "SUBJECT", "SUBJECT_NOT_SET")
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
    prior_kappa_mean = get(hyperparam_dict, "prior_kappa_mean", "NOT SET")
    prior_kappa_variance = get(hyperparam_dict, "prior_kappa_variance", "NOT SET")
    prior_omega_mean = get(hyperparam_dict, "prior_omega_mean", "NOT SET")
    prior_omega_variance = get(hyperparam_dict, "prior_omega_variance", "NOT SET")
    prior_beta_shape = get(hyperparam_dict, "prior_beta_shape", "NOT SET")
    prior_beta_rate = get(hyperparam_dict, "prior_beta_rate", "NOT SET")
    prior_z_prev_mean = get(hyperparam_dict, "prior_z_prev_mean", "NOT SET")
    prior_z_prev_variance = get(hyperparam_dict, "prior_z_prev_variance", "NOT SET")
    prior_x_prev_mean = get(hyperparam_dict, "prior_x_prev_mean", "NOT SET")
    prior_x_prev_variance = get(hyperparam_dict, "prior_x_prev_variance", "NOT SET")
    initial_z_mean = get(hyperparam_dict, "initial_z_mean", "NOT SET")
    initial_z_variance = get(hyperparam_dict, "initial_z_variance", "NOT SET")
    initial_kappa_mean = get(hyperparam_dict, "initial_kappa_mean", "NOT SET")
    initial_kappa_variance = get(hyperparam_dict, "initial_kappa_variance", "NOT SET")
    initial_omega_mean = get(hyperparam_dict, "initial_omega_mean", "NOT SET")
    initial_omega_variance = get(hyperparam_dict, "initial_omega_variance", "NOT SET")
    initial_beta_shape = get(hyperparam_dict, "initial_beta_shape", "NOT SET")
    initial_beta_rate = get(hyperparam_dict, "initial_beta_rate", "NOT SET")
    niterations = get(hyperparam_dict, "niterations", "NOT SET")
    niterations = Int(niterations) # cast as an integer
else
    println("Running locally...")
    Pkg.activate("emotional_faces_scripts/"); # Note that the Project and Manifest files are in the same directory as this script.
    # Click Julia: Activate this Environment to run the REPL
    subject_id = "5a5ec79cacc75b00017aa095"
    predictions_or_responses = "responses" # Haven't set up infrastructure to fit predictions
    results_dir = "outputs"
    root = ""
    # get current datetime
    using Dates
    now_time = now()
    formatted_time = "_" * Dates.format(now_time, "yyyy-mm-ddTHH_MM_SS")
    prior_kappa_mean = 1.0
    prior_kappa_variance = 0.1
    prior_omega_mean = 0.0
    prior_omega_variance = .01 # used to be .01
    prior_beta_shape = 1.0
    prior_beta_rate = 1.0
    prior_z_prev_mean = 0.0
    prior_z_prev_variance = 1.0
    prior_x_prev_mean = 0.0
    prior_x_prev_variance = 1.0
    initial_z_mean = 0.0
    initial_z_variance = 1.0
    initial_kappa_mean = 1.0
    initial_kappa_variance = 0.1
    initial_omega_mean = 0.0
    initial_omega_variance = 0.01
    initial_beta_shape = 0.1
    initial_beta_rate = 0.1
    niterations = 20
end
println("--------------------")
println("Subject ID: ", subject_id)
println("Predictions or Responses: ", predictions_or_responses)
println("Results Directory: ", results_dir)
println("Root Directory: ", root)
println("Formatted Time: ", formatted_time)
println("Prior Kappa Mean: ", prior_kappa_mean)
println("Prior Kappa Variance: ", prior_kappa_variance)
println("Prior Omega Mean: ", prior_omega_mean)
println("Prior Omega Variance: ", prior_omega_variance)
println("Prior Beta Shape: ", prior_beta_shape)
println("Prior Beta Rate: ", prior_beta_rate)
println("Prior Z Previous Mean: ", prior_z_prev_mean)
println("Prior Z Previous Variance: ", prior_z_prev_variance)
println("Prior X Previous Mean: ", prior_x_prev_mean)
println("Prior X Previous Variance: ", prior_x_prev_variance)
println("Initial Z Mean: ", initial_z_mean)
println("Initial Z Variance: ", initial_z_variance)
println("Initial Kappa Mean: ", initial_kappa_mean)
println("Initial Kappa Variance: ", initial_kappa_variance)
println("Initial Omega Mean: ", initial_omega_mean)
println("Initial Omega Variance: ", initial_omega_variance)
println("Initial Beta Shape: ", initial_beta_shape)
println("Initial Beta Rate: ", initial_beta_rate)
println("Number of iterations: ", niterations)
println("--------------------")
ENV["JULIA_PKG_SERVER"] = "https://pkg.julialang.org"
println("Loading packages...")
Pkg.instantiate()  # Reinstall missing dependencies
println("Packages loaded.")


using RxInfer
using ExponentialFamilyProjection, ExponentialFamily
using Distributions
using Plots, StatsPlots
using StableRNGs
using CSV, DataFrames

# Import file containing callbacks functions and tagged logger
include(root * "callbacks.jl")

# Seed for reproducibility
seed = 23
rng = StableRNG(seed)
## Feng: rng is not used in the code, how to affect the infer always have the same result???

# Read in the task data
file_name = root * "emotional_faces_processed_data/task_data_$(subject_id)_$(predictions_or_responses).csv"
data = CSV.read(file_name, DataFrame)
# Create variable for observations
obs_data = data.observed 
obs_data = Float64.(obs_data)
# Create variable for responses
resp_data = data.response
resp_data = Float64.(resp_data)

# For debugging, let's just use two observations and responses
resp_data = resp_data[1:2]
obs_data = obs_data[1:2]

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




# todo: esitmate variance on top level (z_variance). could potentially fix the middle mean to .5; 
@model function hgf_smoothing(obs, resp)
    # Initial states - adjust means to be closer to data range
    # z_variance ~ Gamma(shape = 1.0, rate = 1.0)  # Reduced variance
    z_initial ~ Normal(mean = prior_z_prev_mean, variance = prior_z_prev_variance)  where { pipeline = TaggedLogger("z_initial") }# Reduced variance
    x_initial ~ Normal(mean = prior_x_prev_mean, variance = prior_x_prev_variance)  where { pipeline = TaggedLogger("x_initial") }# Reduced variance
    
    x_prev = x_initial 
    z_prev = z_initial 

    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = prior_kappa_mean, variance = prior_kappa_variance)  where { pipeline = TaggedLogger("κ") }# Reduced variance and mean
    ω ~ Normal(mean = prior_omega_mean, variance = prior_omega_variance)  where { pipeline = TaggedLogger("ω") }# Reduced variance
    β ~ Gamma(shape = prior_beta_shape, rate = prior_beta_rate)  where { pipeline = TaggedLogger("β") } # Less constrained

    for i in eachindex(obs)
        # Higher layer update (Gaussian random walk)
        z[i] ~ Normal(mean = z_prev, variance = 1)  where { pipeline = TaggedLogger("z[$(i)]") }
 
        # Lower layer update
        x[i] ~ GCV(x_prev, z[i], κ, ω) where { pipeline = TaggedLogger("x[$(i)]") }

        # Noisy binary observations (Bernoulli likelihood)
        obs[i] ~ Probit(x[i]) where { pipeline = TaggedLogger("obs[$(i)]") }
 
        # Noisy binary response (Bernoulli likelihood)
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
    #q(x, temp, z, κ, ω, β, x_initial,z_variance) = q(x, temp, x_initial)q(z,z_variance)q(κ)q(ω)q(β)
    q(x, temp, z, κ, ω, β, x_initial) = q(x, temp, x_initial)q(z)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
   # q(z_variance) :: ProjectedTo(Gamma)    
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
        # μ(z) = NormalMeanVariance(initial_z_mean, initial_z_variance)
        q(z) = NormalMeanVariance(initial_z_mean, .02)
        q(κ) = NormalMeanVariance(initial_kappa_mean, .03)
        q(ω) = NormalMeanVariance(initial_omega_mean, .04)
        q(β) = GammaShapeRate(initial_beta_shape, .05)
        # q(z_variance) = GammaShapeRate(1, 1)
        # μ(z) = map(_ -> NormalMeanPrecision(0, 1), 1:200) # create a vector of 200 Normal distributions
        # μ(x) = map(_ -> NormalMeanPrecision(0, 1), 1:200) # create a vector of 200 Normal distributions
        # μ(z) = NormalMeanPrecision(0, 1)
        # μ(x) =NormalMeanPrecision(0, 1)
        
        #q(z_prev) = NormalMeanVariance(initial_z_mean, initial_z_variance)
    end

    return infer(
        model = hgf_smoothing(),  # Reduced z_variance
        data = (obs = obs, resp = UnfactorizedData(resp),),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = niterations,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (x = KeepEach(), z = KeepEach(), ω=KeepEach(), κ=KeepEach(), β=KeepEach(), temp=KeepEach(),),
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
local infer_result
if get(ENV, "CLUSTER", "false") == "true"
    infer_result = run_inference_smoothing(obs_data, resp_data, niterations)
else
    # Open a log file with w, meaning it gets overwritten each time
    println("Logging to file...")
    logfile = open(joinpath(pwd(), "outputs", "inference.log"), "w")
    
    infer_result = redirect_stdout(logfile) do
        redirect_stderr(logfile) do
            run_inference_smoothing(obs_data, resp_data, niterations)
        end
    end   # infer_result now holds whatever run_inference_smoothing returned
    
    close(logfile)
end


# If free energy was computed, find the last iteration where free energy was neither NaN or infinity
if compute_bethe_free_energy
    free_energy_vals = infer_result.free_energy
    niterations_used = findlast(isfinite, free_energy_vals) # checks for not NaN or inf
else
    niterations_used = niterations
end

# Extract posteriors
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
temp_posterior = infer_result.posteriors[:temp][niterations_used]
temp_posterior = getfield.(temp_posterior, :data) # added this line because of addons

# Print mean and standard deviation of parameters
println("Parameter Estimates:")
println("κ: mean = $(mean(κ_posterior)), std = $(std(κ_posterior))")
println("ω: mean = $(mean(ω_posterior)), std = $(std(ω_posterior))")
println("β: mean = $(mean(β_posterior)), std = $(std(β_posterior))")

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
    layout=(1,3)
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
    beta_mean = mean(β_posterior),
    beta_std = std(β_posterior),
    prior_kappa_mean = prior_kappa_mean,
    prior_kappa_variance = prior_kappa_variance,
    prior_omega_mean = prior_omega_mean,
    prior_omega_variance = prior_omega_variance,
    prior_beta_shape = prior_beta_shape,
    prior_beta_rate = prior_beta_rate,
    prior_z_prev_mean = prior_z_prev_mean,
    prior_z_prev_variance = prior_z_prev_variance,
    prior_x_prev_mean = prior_x_prev_mean,
    prior_x_prev_variance = prior_x_prev_variance,
    initial_z_mean = initial_z_mean,
    initial_z_variance = initial_z_variance,
    initial_kappa_mean = initial_kappa_mean,
    initial_kappa_variance = initial_kappa_variance,
    initial_omega_mean = initial_omega_mean,
    initial_omega_variance = initial_omega_variance,
    initial_beta_shape = initial_beta_shape,
    initial_beta_rate = initial_beta_rate,
    niterations = niterations,
)



# Save the results as a CSV 
output_file = joinpath(results_dir, "model_results_" * subject_id * formatted_time * ".csv")
CSV.write(output_file, results_df)
