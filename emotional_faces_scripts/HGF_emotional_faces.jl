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
    prior_z_mean = get(hyperparam_dict, "prior_z_mean", "NOT SET")
    prior_z_variance = get(hyperparam_dict, "prior_z_variance", "NOT SET")
    prior_x_mean = get(hyperparam_dict, "prior_x_mean", "NOT SET")
    prior_x_variance = get(hyperparam_dict, "prior_x_variance", "NOT SET")
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
    Pkg.activate("L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/"); # Note that the Project and Manifest files are in the same directory as this script.
    # Click Julia: Activate this Environment to run the REPL
    subject_id = "5a5ec79cacc75b00017aa095"
    subject_id = "6630fcae418bde6a9e7c3b47"
    predictions_or_responses = "responses" # Haven't set up infrastructure to fit predictions
    results_dir = "L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_test"
    root = "L:/"
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
    prior_z_mean = 0.0
    prior_z_variance = 1.0
    prior_x_mean = 0.0
    prior_x_variance = 1.0
    initial_z_mean = 0.0
    initial_z_variance = 10.0
    initial_kappa_mean = 1.0
    initial_kappa_variance = 0.1
    initial_omega_mean = 0.0
    initial_omega_variance = 0.01
    initial_beta_shape = 0.1
    initial_beta_rate = 0.1
    niterations = 20
end
Pkg.instantiate()  # Reinstall missing dependencies


using RxInfer
using ExponentialFamilyProjection, ExponentialFamily
using Distributions
using Plots, StatsPlots
using StableRNGs
using CSV, DataFrames


# Seed for reproducibility
seed = 23
rng = StableRNG(seed)

# Read in the task data
file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_processed_data/task_data_$(subject_id)_$(predictions_or_responses).csv"
data = CSV.read(file_name, DataFrame)
# Create variable for observations
obs_data = data.observed 
obs_data = Float64.(obs_data)
# Create variable for responses
resp_data = data.response
resp_data = Float64.(resp_data)
# Count the number of missing responses
nan_responses = count(isnan, resp_data)
println("Number of NaN Responses: ", nan_responses)
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
@model function hgf_smoothing(obs, resp, z_variance)
    # Initial states - adjust means to be closer to data range
    z_prev ~ Normal(mean = prior_z_mean, variance = prior_z_variance)  # Reduced variance
    x_prev ~ Normal(mean = prior_x_mean, variance = prior_x_variance)  # Reduced variance
 
    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = prior_kappa_mean, variance = prior_kappa_variance)  # Reduced variance and mean
    ω ~ Normal(mean = prior_omega_mean, variance = prior_omega_variance)  # Reduced variance
    β ~ Gamma(shape = prior_beta_shape, rate = prior_beta_rate)  # Less constrained

    for i in eachindex(obs)
        # Higher layer update (Gaussian random walk)
        z[i] ~ Normal(mean = z_prev, variance = z_variance)
 
        # Lower layer update
        x[i] ~ GCV(x_prev, z[i], κ, ω)

        # Noisy binary observations (Bernoulli likelihood)
        obs[i] ~ Probit(x[i])
 
        # Noisy binary response (Bernoulli likelihood)
        temp[i] ~ softdot(β, x[i], 1.0)
        resp[i] ~ Probit(temp[i])

        # Failed Solution 1:
        # if !ismissing(resp[i])
        #     temp[i] ~ softdot(β, x[i], 1.0)
        #     resp[i] ~ Probit(temp[i])
        # end

        # Failed Solution 2:
        # if !ismissing(resp[i])
        #     temp[i] ~ softdot(β, x[i], 1.0)
        #     resp[i] ~ Probit(temp[i])
        # else
        #     temp[i] ~ Normal(mean = 0.0, precision = 1.0)  # Dummy placeholder
        #     resp[i] ~ Normal(mean = temp[i], precision = 1.0)  # Dummy placeholder
        # end

        # Failed Solution 3:
        #  if !ismissing(resp[i])
        #     temp[i] ~ softdot(β, x[i], 1.0)
        #     resp[i] ~ Probit(temp[i])
        # else
        #     temp[i] ~ Uninformative()  # Dummy placeholder
        #     resp[i] ~ Probit(temp[i])  # Dummy placeholder
        # end

        # Update previous states
        z_prev = z[i]
        x_prev = x[i]
    end 
 end

 #  # Add a rule to account for missing input to the Probit node
#  # Todo - might want to change this
#  @rule Probit(:out, Marginalisation) (q_in::Missing, meta::ProbitMeta) = begin
#     return missing
# end

# # @rule Probit(:out, Marginalisation) (q_in::Uninformative, meta::ProbitMeta) = begin
#     return Bernoulli(0.5)
# end

# @rule Probit(:out, Marginalisation) (q_in::NormalMeanVariance, meta::ProbitMeta) = begin
#     return Bernoulli(0.5)
# end

# @rule Probit(:out, Marginalisation) (q_in::NormalWeightedMeanPrecision, meta::ProbitMeta) = begin
#     return Bernoulli(0.5)
# end
 
@constraints function hgfconstraints_smoothing() 
    #Structured mean-field factorization constraints
    q(x_prev, x, temp, z, κ, ω, β) = q(x_prev, x, temp)q(z)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
end

import ReactiveMP: as_companion_matrix, ar_transition, getvform, getorder, add_transition!
# Add Base methods to prevent compiler ambiguity
Base.prod(::GenericProd, left::ProductOf{L, R}, ::Missing) where {L, R} = left


@meta function hgfmeta_smoothing()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV() -> GCVMetadata(GaussHermiteCubature(31))
end



function before_model_creation()
    println("The model is about to be created")
end

function after_model_creation(model::ProbabilisticModel)
    println("The model has been created")
    println("  The number of factor nodes is: ", length(RxInfer.getfactornodes(model)))
    println("  The number of latent states is: ", length(RxInfer.getrandomvars(model)))
    println("  The number of data points is: ", length(RxInfer.getdatavars(model)))
    println("  The number of constants is: ", length(RxInfer.getconstantvars(model)))
end

function before_inference(model::ProbabilisticModel)
    println("The inference procedure is about to start")
end

function after_inference(model::ProbabilisticModel)
    println("The inference procedure has ended")
end

function before_iteration(model::ProbabilisticModel, iteration::Int)
    println("The iteration ", iteration, " is about to start")
end

function after_iteration(model::ProbabilisticModel, iteration::Int)
    println("The iteration ", iteration, " has ended")
end

function before_data_update(model::ProbabilisticModel, data)
    println("The data is about to be processed")
end

function after_data_update(model::ProbabilisticModel, data)
    println("The data has been processed")
end

function on_marginal_update(model::ProbabilisticModel, name, update)
    # Check if the name is :x or :ω (Symbols)
    if name == :κ || name == :ω
        println("New marginal update for ", name, ": ", update)
    end
end





function run_inference_smoothing(obs, resp, z_variance, niterations)
    @initialization function hgf_init_smoothing()
        q(z) = NormalMeanVariance(initial_z_mean, initial_z_variance)
        q(κ) = NormalMeanVariance(initial_kappa_mean, initial_kappa_variance)
        q(ω) = NormalMeanVariance(initial_omega_mean, initial_omega_variance)
        q(β) = GammaShapeRate(initial_beta_shape, initial_beta_rate)
    end

    return infer(
        model = hgf_smoothing(z_variance=z_variance,),  # Reduced z_variance
        data = (obs = obs, resp = UnfactorizedData(resp),),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = niterations,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (x = KeepLast(), z = KeepLast(), ω=KeepLast(), κ=KeepLast(), β=KeepLast(), temp=KeepLast(),),
        free_energy = compute_bethe_free_energy,  # Compute Bethe Free Energy when there are no missing values
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
    )
end

infer_result = run_inference_smoothing(obs_data, resp_data, 1.0, niterations)

# Extract posteriors
x_posterior = infer_result.posteriors[:x]
z_posterior = infer_result.posteriors[:z]
κ_posterior = infer_result.posteriors[:κ] 
ω_posterior = infer_result.posteriors[:ω]
β_posterior = infer_result.posteriors[:β]
temp_posterior = infer_result.posteriors[:temp]
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



## save results
results_df = DataFrame(
    id = subject_id,
    number_missing_responses = nan_responses,
    kappa_mean = mean(κ_posterior),
    kappa_std = std(κ_posterior),
    omega_mean = mean(ω_posterior),
    omega_std = std(ω_posterior),
    beta_mean = mean(β_posterior),
    beta_std = std(β_posterior),
    free_energy_last = last(free_energy_vals),
    mean_action_prob = mean(action_probability_values),
    mean_model_accuracy = mean_model_accuracy,
    prior_kappa_mean = prior_kappa_mean,
    prior_kappa_variance = prior_kappa_variance,
    prior_omega_mean = prior_omega_mean,
    prior_omega_variance = prior_omega_variance,
    prior_beta_shape = prior_beta_shape,
    prior_beta_rate = prior_beta_rate,
    prior_z_mean = prior_z_mean,
    prior_z_variance = prior_z_variance,
    prior_x_mean = prior_x_mean,
    prior_x_variance = prior_x_variance,
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
