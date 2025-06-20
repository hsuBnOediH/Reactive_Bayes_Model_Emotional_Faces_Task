using SequentialSamplingModels
using Plots
using Random

Random.seed!(8741)
ν= -8.0
α = 1.0
τ = 0.30
z = 0.50
dist = DDM(ν, α, τ, z)
choices,rts = rand(dist, 10_000)
choices
#count the number of choice ==1 and choice == 2
count(choices .== 1), count(choices .== 2)
#plot the bar of number of choices
bar([1, 2], [count(choices .== 1), count(choices .== 2)], xlabel="Choice", ylabel="Count", title="Choices Distribution")