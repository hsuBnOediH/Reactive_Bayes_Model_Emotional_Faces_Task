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

#####################################################
 #  try learning the parameters k and w
 #  Model for offline learning (smoothing)
 #  Model for offline learning (smoothing)
 @model function hgf_smoothing(y, z_variance, y_variance)
    # Initial states 
    z_prev ~ Normal(mean = 0., variance = 5.0)
    x_prev ~ Normal(mean = 0., variance = 5.0)

    # Priors on κ and ω
    κ ~ Normal(mean = 1.5, variance = 1.0)
    ω ~ Normal(mean = 0.0, variance = 0.05)

    for i in eachindex(y)
        # Higher layer 
        z[i] ~ Normal(mean = z_prev, variance = z_variance)

        # Lower layer 
        x[i] ~ GCV(x_prev, z[i], κ, ω)

        # Noisy observations 
        y[i] ~ Normal(mean = x[i], variance = y_variance)

        # Update last/previous hidden states
        z_prev = z[i]
        x_prev = x[i]
    end
end

@constraints function hgfconstraints_smoothing() 
    #Structured mean-field factorization constraints
    q(x_prev,x, z,κ,ω) = q(x_prev,x)q(z)q(κ)q(ω)
end

@meta function hgfmeta_smoothing()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV() -> GCVMetadata(GaussHermiteCubature(31)) 
end


function run_inference_smoothing(data, z_variance, y_variance)
    @initialization function hgf_init_smoothing()
        q(x) = NormalMeanVariance(0.0,5.0)
        q(z) = NormalMeanVariance(0.0,5.0)
        q(κ) = NormalMeanVariance(1.5,1.0)
        q(ω) = NormalMeanVariance(0.0,0.05)
    end

    #Let's do inference with 20 iterations 
    return infer(
        model = hgf_smoothing(z_variance = z_variance, y_variance = y_variance,),
        data = (y = data,),
        meta = hgfmeta_smoothing(),
        constraints = hgfconstraints_smoothing(),
        initialization = hgf_init_smoothing(),
        iterations = 20,
        options = (limit_stack_depth = 100, ), 
        returnvars = (x = KeepLast(), z = KeepLast(),ω=KeepLast(),κ=KeepLast(),),
        free_energy = true 
    )
end

# Read the CSV file
data = CSV.read("task_data_5a5ec79cacc75b00017aa095.csv", DataFrame)

# Extract the 'observed' column as a vector
y = data.observed

# Convert to Float64 if needed (since your HGF model expects floating point values)
y = Float64.(y)

# You can verify the data was loaded correctly
# println("First 10 observations: ", y[1:10])
# println("Length of data: ", length(y))
z_variance = abs2(0.2)
y_variance = abs2(.4)
result_smoothing = run_inference_smoothing(y, z_variance, y_variance);
mz_smoothing = result_smoothing.posteriors[:z];
mx_smoothing = result_smoothing.posteriors[:x];

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
plot(result_smoothing.free_energy, label = "Bethe Free Energy")

# plot the posteriors of κ and ω
q_κ = result_smoothing.posteriors[:κ]
q_ω = result_smoothing.posteriors[:ω]
# q_y_variance = result_smoothing.posteriors[:y_variance] 

println("Approximate value of κ: ", mean(q_κ))
println("Approximate value of ω: ", mean(q_ω))
# println("Approximate value of y_variance: ", mean(q_y_variance))

# visualize the marginal posteriors of κ and ω
range_w = range(-1,0.5,length = 200)
range_k = range(0,2,length = 200)
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
