# import Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.activate(".")  # Activate the current directory as the project environment
# Pkg.add("RxInfer")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("ReactiveMP")
# Pkg.add("Cairo")
# Pkg.add("GraphPlot")
# Pkg.add("Plots")
# Pkg.add("BenchmarkTools")
# Pkg.add("StableRNGs")
# Pkg.add("ExponentialFamilyProjection")
# Pkg.add("ExponentialFamily")
# Pkg.add("StatsPlots")
# Pkg.add("ReactiveMP")
# Activate local environment, see `Project.toml`
import Pkg; Pkg.activate("."); Pkg.instantiate();

using RxInfer
using ExponentialFamilyProjection, ExponentialFamily
using Distributions
using Plots, StatsPlots
using StableRNGs


# Seed for reproducibility
seed = 23
rng = StableRNG(seed)


# Generate 30 trials of data that could represent a learning task
# where the correct response pattern changes over time
n_trials = 201

# Create a pattern where the correct response changes a few times
# This simulates a learning environment where rules change
function generate_task_data(n_trials)
    # Initialize arrays
    obs = zeros(n_trials)
    resp = zeros(n_trials)
    
    # Create blocks of 100 trials each
    block_size = 100
    n_blocks = ceil(Int, n_trials/block_size)
    
    current_idx = 1
    for block in 1:n_blocks
        if current_idx > n_trials
            break
        end
        
        end_idx = min(current_idx + block_size - 1, n_trials)
        
        # Alternate between different patterns
        pattern_type = mod(block, 3)
        if pattern_type == 0
            # Block type 1: mostly 1s
            obs[current_idx:end_idx] .= rand([1.0, 1.0, 1.0, 0.0], end_idx - current_idx + 1)
        elseif pattern_type == 1
            # Block type 2: mostly 0s
            obs[current_idx:end_idx] .= rand([0.0, 0.0, 0.0, 1.0], end_idx - current_idx + 1)
        else
            # Block type 3: alternating with noise
            base_pattern = repeat([1.0, 0.0], ceil(Int, (end_idx - current_idx + 1)/2))
            obs[current_idx:end_idx] .= base_pattern[1:(end_idx - current_idx + 1)]
            # Add some noise
            noise_idx = rand(1:(end_idx - current_idx + 1), ceil(Int, (end_idx - current_idx + 1)*0.1))
            obs[current_idx .+ noise_idx .- 1] .= 1.0 .- obs[current_idx .+ noise_idx .- 1]
        end
        
        # Generate responses with learning
        error_prob = 0.4 * exp(-block/5) # Exponentially decreasing error rate
        for i in current_idx:end_idx
            if rand() < error_prob
                resp[i] = 1.0 - obs[i]  # Wrong response
            else
                resp[i] = obs[i]  # Correct response
            end
        end
        
        current_idx += block_size
    end
    
    return obs, resp
end

# Generate the data
obs_data, resp_data = generate_task_data(n_trials)

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
        iterations = 30,  # More iterations
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

# Extract free energy values
free_energy_vals = infer_result.free_energy
println("Free Energy values: ", free_energy_vals)

# Plot Free Energy to check convergence
plot(free_energy_vals, title="Free Energy over Iterations", xlabel="Iteration", ylabel="Free Energy", label="")