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
# Activate local environment, see `Project.toml`
import Pkg; Pkg.activate("."); Pkg.instantiate();

using RxInfer, BenchmarkTools, Random, Plots, StableRNGs, CSV, DataFrames


# Seed for reproducibility
seed = 23
rng = StableRNG(seed)


# define nonlinear node
# function sigmoid(x)
#     return 1.0 / (1.0 + exp(-x))
# end

# dot([2,3],[4,5])
# dot([5,2],6)

#####################################################
 #  try learning the parameters k and w
 #  Model for offline learning (smoothing)
 @model function hgf_smoothing(obs, resp, z_variance)
    # Initial states
    z_prev ~ Normal(mean = 0.0, variance = 1.0)  # Higher layer hidden state
    x_prev ~ Normal(mean = 0.0, variance = 1.0)  # Lower layer hidden state

    # Priors on κ and ω
    κ ~ Normal(mean = 1.0, variance = 1.0)
    ω ~ Normal(mean = 0.0, variance = 1.0)
    β ~ Gamma(shape = 2.0, rate = 2.0)  # Inverse temperature parameter for response model

    for i in eachindex(obs)
        # Higher layer update (Gaussian random walk)
        z[i] ~ Normal(mean = z_prev, variance = z_variance)

        # Lower layer update
        x[i] ~ GCV(x_prev, z[i], κ, ω)

        # Apply sigmoid function to convert to probability
        # p[i] := sigmoid(x[i]) # Sigmoid transformation

        # Noisy binary observations (Bernoulli likelihood)
        # obs[i] ~ Bernoulli(p[i])

        # use probit function to convert continuous value x to binary outcome obs
        obs[i] ~ Probit(x[i])

        # Response model: assume a softmax function governs responses
        # resp_p[i] := sigmoid(β * x[i])

        # use softdot function to govern temperature
        # temp = obs[i]
        resp_p[i] ~ SoftDot(x[i], β, 1e2)


        # Noisy binary response (Bernoulli likelihood)
        resp[i] ~ Probit(resp_p[i])

        # Update previous states
        z_prev = z[i]
        x_prev = x[i]
    end 
end


@constraints function hgfconstraints_smoothing() 
    #Structured mean-field factorization constraints
    # q(x_prev,x, z,κ,ω,β) = q(x_prev,x)q(z)q(κ)q(ω)q(β)
    q(x_prev,x, z,κ,ω,β) = q(x_prev,x,β)q(z)q(κ)q(ω)
    # q(β) :: PointMassFormConstraint()
    # q(β) :: SampleListFormConstraint(1000)

end

@meta function hgfmeta_smoothing()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV() -> GCVMetadata(GaussHermiteCubature(31)) 
    # sigmoid() -> DeltaMeta(method = Linearization())
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





function run_inference_smoothing(obs, resp, z_variance)
    @initialization function hgf_init_smoothing()
        q(x) = NormalMeanVariance(0.0,1.0)
        μ(x) = vague(NormalMeanVariance)
        q(z) = NormalMeanVariance(0.0,1.0)
        q(κ) = NormalMeanVariance(1.0,1.0)
        q(ω) = NormalMeanVariance(0.0,1.0)
        q(β) = GammaShapeRate(2.0,2.0)
    end

    #Let's do inference with 20 iterations 
    return infer(
        model = hgf_smoothing(z_variance = z_variance,),
        data = (obs = obs,resp=resp,),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = 20,
        options = (limit_stack_depth = 100,), 
        returnvars = (x = KeepLast(), z = KeepLast(),ω=KeepLast(),κ=KeepLast(),β=KeepLast(),),
        free_energy = true,
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
        )
    )
end

# Read the CSV file
data = CSV.read("task_data_5a5ec79cacc75b00017aa095.csv", DataFrame)

# Extract the 'observed' column as a vector
obs = data.observed
# Convert to Float64 
obs = Float64.(obs)

# Extract the 'response' column as a vector
resp = data.response
# Convert to Float64
resp = Float64.(resp)




# You can verify the data was loaded correctly
# println("First 10 observations: ", obs[1:10])
# println("Length of data: ", length(obs))
z_variance = abs2(0.2)

result_smoothing = run_inference_smoothing(obs, resp, z_variance);
mz_smoothing = result_smoothing.posteriors[:z];
mx_smoothing = result_smoothing.posteriors[:x];
n = length(obs)  
let 
    pz = plot(title = "Hidden States Z")
    px = plot(title = "Hidden States X")
    
    # plot!(pz, 1:n, z, label = "z_i", color = :orange)
    plot!(pz, 1:n, mean.(mz_smoothing), ribbon = std.(mz_smoothing), label = "estimated z_i", color = :teal)
    
    # plot!(px, 1:n, x, label = "x_i", color = :green)
    plot!(px, 1:n, mean.(mx_smoothing), ribbon = std.(mx_smoothing), label = "estimated x_i", color = :violet)
    
    plot(pz, px, layout = @layout([ a; b ]))
    savefig("EF_model_results_smoothing.png")
end

# plot the free energy
let
    plot(result_smoothing.free_energy, label = "Bethe Free Energy")
    savefig("Bethe_free_energy.png")
end

# plot the posteriors of κ and ω
q_κ = result_smoothing.posteriors[:κ]
q_ω = result_smoothing.posteriors[:ω]

println("Approximate value of κ: ", mean(q_κ))
println("Approximate value of ω: ", mean(q_ω))

# visualize the marginal posteriors of κ and ω
range_w = range(-1,0.5,length = 200)
range_k = range(0,6,length = 200)
let 
    pw = plot(title = "Marginal q(w)")
    pk = plot(title = "Marginal q(k)")
    
    plot!(pw, range_w, (x) -> pdf(q_ω, x), fillalpha=0.3, fillrange = 0, label="Posterior q(w)", c=3, legend_position=(0.1,0.95), legendfontsize=9)
    # vline!([real_w], label="Real w")
    xlabel!("w")
    
    
    plot!(pk, range_k, (x) -> pdf(q_κ, x), fillalpha=0.3, fillrange = 0, label="Posterior q(k)", c=3, legend_position=(0.1,0.95), legendfontsize=9)
    # vline!([real_k], label="Real k")
    xlabel!("k")
    
    plot(pk, pw, layout = @layout([ a; b ]))
    savefig("EF_marginal_posteriors.png")
end
