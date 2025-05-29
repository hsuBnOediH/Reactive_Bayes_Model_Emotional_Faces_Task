using SequentialSamplingModels
using Plots
using Random

Random.seed!(2054)

# Create LBA distribution with known parameters
dist = LBA(; ν=[2.75,1.75], A=0.8, k=0.5, τ=0.25)

# Sample 10,000 simulated data from the LBA
sim_data = rand(dist, 10_000)
# compute log likelihood of simulated data
logpdf(dist, sim_data)

# Plot the RT distribution for each choice
histogram(dist)

plot!(dist; t_range=range(.3,2.5, length=100), xlims=(0, 2.5))