using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
Random.TaskLocalRNG()

# Typical range: -5 < ν < 5
ν=1.0
# Large values indicates response caution. Typical range: 0.5 < α < 2
α = 0.80
# non-decisional processes . Typical range: 0.1 < τ < 0.5
τ = 0.30
# An indicator of an an initial bias towards a decision.
z = 0.10

dist = DDM(ν, α, τ, z)

# choices,rts = rand(dist, 10_000)
choices,rts = rand(dist, 10)
# count how many times each choice is 1 and 2

pdf_res = pdf.(dist, choices, rts)

cdf(dist, 1, 10)

histogram(dist) 
plot!(dist; t_range=range(.301, 1, length=100))

