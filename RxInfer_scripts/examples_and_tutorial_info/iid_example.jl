# `ExponentialFamily` package expors different parametrizations
# for the Normal distribution
using ExponentialFamily
using StableRNGs
using RxInfer
using Distributions
using Plots, StatsPlots

hidden_μ       = 3.1415
hidden_τ       = 2.7182
distribution   = NormalMeanPrecision(hidden_μ, hidden_τ)
rng            = StableRNG(42)
n_observations = 5
dataset        = rand(rng, distribution, n_observations)


@model function iid_estimation(y)
    μ  ~ Normal(mean = 0.0, precision = 0.1)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end
# Specify mean-field constraint over the joint variational posterior
constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)
end


@model function iid_estimation(y)
    μ  ~ Normal(mean = 0.0, precision = 0.1)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    x ~ Normal(mean = μ, precision = τ)
    y .~ Normal(mean = x, precision = 2)
end



# Specify mean-field constraint over the joint variational posterior
constraints = @constraints begin
    q(μ, τ, x) = q(μ)q(τ)q(x)
end
# Specify initial posteriors for variational iterations
initialization = @initialization begin
    q(μ) = vague(NormalMeanPrecision)
    q(τ) = vague(GammaShapeRate)
    q(x) = vague(NormalMeanPrecision)
end

results = infer(
    model = iid_estimation(),
    data  = (y = dataset, ),
    initialization = initialization
)