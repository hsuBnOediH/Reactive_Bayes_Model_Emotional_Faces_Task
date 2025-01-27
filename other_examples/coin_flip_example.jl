# uncomment the code below if running for the first time to install the packages
import Pkg
# Pkg.add("RxInfer")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("ReactiveMP")
# Pkg.add("Cairo")
# Pkg.add("GraphPlot")
# Pkg.add("GraphPlot")
# Pkg.add("Plots")

using RxInfer, Distributions, Random

# Random number generator for reproducibility
rng            = MersenneTwister(42)
# Number of coin flips (observations)
n_observations = 10
# The bias of a coin used in the demonstration
coin_bias      = 0.75
# We assume that the outcome of each coin flip is
# distributed as the `Bernoulli` distrinution
distribution   = Bernoulli(coin_bias)
# Simulated coin flips
dataset        = rand(rng, distribution, n_observations)

display(dataset)

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(y, a, b)
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

conditioned = coin_model(a = 2.0, b = 7.0) | (y = [ true, false, true ], )
# `Create` the actual graph of the model conditioned on the data
model = RxInfer.create_model(conditioned)
# Call `gplot` function from `GraphPlot` to visualise the structure of the graph
using Cairo, GraphPlot, GraphPPL
GraphPlot.gplot(RxInfer.getmodel(model))
result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = dataset, )
)
estimated_theta = result.posteriors[:θ]
println("Real bias is ", coin_bias)
println("Estimated bias is ", mean(estimated_theta))
println("Standard deviation ", std(estimated_theta))
using Plots

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)

plot(p1, p2, layout = @layout([ a; b ]))





# we're also able to condition on data that is not available at model creation
# The only difference here is that we do not specify `a` and `b` as hyperparameters
# But rather indicate that the data for them will be available later during the inference
conditioned_with_deffered_data = coin_model() | (
    y = [ true, false, true ],
    a = RxInfer.DeferredDataHandler(),
    b = RxInfer.DeferredDataHandler()
)

# The graph creation API does not change
model_with_deffered_data = RxInfer.create_model(conditioned_with_deffered_data)
# We can visualise the graph with missing data handles as well
GraphPlot.gplot(RxInfer.getmodel(model_with_deffered_data))



model = RxInfer.create_model(conditioned)

