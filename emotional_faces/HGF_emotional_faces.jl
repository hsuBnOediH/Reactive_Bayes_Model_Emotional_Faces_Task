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

    Pkg.activate("/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces/cluster_environment/")
    subject_id = get(ENV, "SUBJECT", "SUBJECT_NOT_SET")
    predictions_or_responses = get(ENV, "PREDICTIONS_OR_RESPONSES","PREDICTIONS_OR_RESPONSES_NOT_SET")
    results_dir = get(ENV, "RESULTS","RESULTS_NOT_SET")
    root = "/media/labs/"
    formatted_time = ""
else
    println("Running locally...")
    Pkg.activate("L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces/"); # Note that the Project and Manifest files are in the same directory as this script
    subject_id = "55bcd160fdf99b1c4e4ae632"
    predictions_or_responses = "responses"
    results_dir = "L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_test"
    root = "L:/"
    # get current datetime
    using Dates
    now_time = now()
    formatted_time = "_" * Dates.format(now_time, "yyyy-mm-ddTHH_MM_SS")
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

file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/data/task_data_$(subject_id)_$(predictions_or_responses).csv"
data = CSV.read(file_name, DataFrame)
obs_data = data.observed
obs_data = Float64.(obs_data)
resp_data = data.response
resp_data = Float64.(resp_data)



# todo: esitmate variance on top level (z_variance). could potentially fix the middle mean to .5; 
@model function hgf_smoothing(obs, resp, z_variance)
    # Initial states - adjust means to be closer to data range
    z_prev ~ Normal(mean = 0.0, variance = 1.0)  # Reduced variance
    x_prev ~ Normal(mean = 0.0, variance = 1.0)  # Reduced variance
 
    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = 1.0, variance = 0.1)  # Reduced variance and mean
    ω ~ Normal(mean = 0.0, variance = 0.01)  # Reduced variance
    β ~ Gamma(shape = 1.0, rate = 1.0)  # Less constrained

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
 
        # Update previous states
        z_prev = z[i]
        x_prev = x[i]
    end 
 end
 
 
@constraints function hgfconstraints_smoothing() 
    #Structured mean-field factorization constraints
    q(x_prev, x, temp, z, κ, ω, β) = q(x_prev, x, temp)q(z)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
end

import ReactiveMP: as_companion_matrix, ar_transition, getvform, getorder, add_transition!

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





function run_inference_smoothing(obs, resp, z_variance, niterations=10)
    @initialization function hgf_init_smoothing()
        q(z) = NormalMeanVariance(0, 10)
        q(κ) = NormalMeanVariance(1.0, 0.1)
        q(ω) = NormalMeanVariance(0.0, 0.01)
        q(β) = GammaShapeRate(0.1, 0.1)
    end

    return infer(
        model = hgf_smoothing(z_variance = 0.1,),  # Reduced z_variance
        data = (obs = obs, resp = resp,),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = niterations,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (x = KeepLast(), z = KeepLast(), ω=KeepLast(), κ=KeepLast(), β=KeepLast(), temp=KeepLast(),),
        free_energy = true,  # Enable to monitor convergence
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

infer_result = run_inference_smoothing(obs_data, resp_data, 5.0, 20)

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

# Extract free energy values
free_energy_vals = infer_result.free_energy
println("Free Energy values: ", free_energy_vals)

# Plot Free Energy to check convergence
plot(free_energy_vals, title="Free Energy over Iterations", xlabel="Iteration", ylabel="Free Energy", label="")
savefig(joinpath(results_dir, "free_energy_" * subject_id * formatted_time * ".png"))


# Get probability assigned to response
# not sure if I should use the mean or precision-weighted mean
probit_prob = [cdf(Normal(0,1), t.xi / t.w) for t in temp_posterior]
probit_prob = [cdf(Normal(0,1), t.xi ) for t in temp_posterior]

action_probability_values = [resp == 1 ? p : 1 - p for (resp, p) in zip(resp_data, probit_prob)]
mean(action_probability_values)






## save results
results_df = DataFrame(
    id = subject_id,
    kappa_mean = mean(κ_posterior),
    kappa_std = std(κ_posterior),
    omega_mean = mean(ω_posterior),
    omega_std = std(ω_posterior),
    beta_mean = mean(β_posterior),
    beta_std = std(β_posterior),
    free_energy_last = last(free_energy_vals),
    mean_action_prob = mean(action_probability_values)
)

# Save the results as a CSV file
output_file = joinpath(results_dir, "model_results_" * subject_id * formatted_time * ".csv")
CSV.write(output_file, results_df)
