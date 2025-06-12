using Distributions
using StatsFuns
using CSV, DataFrames
using StableRNGs
using RxInfer
using BenchmarkTools, Plots

root = "L:/"
subject_id = "AA003" # Use a prolific or local subject id e.g., 5a5ec79cacc75b00017aa095
# Local subject IDs will always be 5 characters, while prolific IDs will always be 24
if length(subject_id) == 5
    study = "local"
elseif length(subject_id) == 24
    study = "prolific"
else
    error("Invalid subject ID length. Must be 5 (local) or 24 (prolific).")
end
if study == "local"
    file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_local/local_emotional_faces_processed_data_06-11-25-18_21_28/task_data_$(subject_id)_processed_data.csv"
elseif study == "prolific"
    file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/prolific_emotional_faces_processed_data_06-11-25-18_23_16/task_data_$(subject_id)_processed_data.csv"
end

data = CSV.read(file_name, DataFrame)
# Create variable for observations which is equal to 1 (congruent, high intensity), .75 (congruent, low intensity), .25 (incongruent, low intensity), 0 (incongruent, high intensity)
#obs_data = data.observed 
obs_data = data.face_intensity
obs_data = Float64.(obs_data)
# Create variable for responses
resp_data = data.resp_sad_high_or_angry_low
resp_data = Float64.(resp_data)

# Participants observe (obs) a high intensity positive signal (1), high intensity negative signal (0), 
# low intensity positive signal (0.75), or a low intensity negative signal (0.25).
# Participants respond (resp) whether they observed a positive or negative signal by responding with a 1 or 0, respectively.




"""
    simulate_hgf_mathys(u, mu0, sa0, kappa, omega, th; irr=Int[], ign=Int[])
Run the 3-level HGF perceptual model on input vector `u`.
- `u`     :: Vector{Float64} external inputs (length n)
- `mu0`   :: Vector{Float64} prior means   (length l)
- `sa0`   :: Vector{Float64} prior variances (length l)
- `kappa`    :: Vector{Float64} phasic couplings (length l−1)
- `omega`    :: Vector{Float64} tonic couplings (length l)
- `z_variance`    :: Float64        top-level volatility
- `irr`   :: Vector{Int}    irregular trials to skip
- `ign`   :: Vector{Int}    trials to ignore  

Returns `(mu, pi)` each ∈ Matrix{Float64}(n×l).
"""
function simulate_hgf_mathys(obs, mu0, sa0, kappa, omega, z_variance; irr=Int[], ign=Int[])
    n, l = length(obs), length(mu0)
    # prepend dummy trial
    obs2 = vcat(0.0, obs)
    t2 = ones(n+1)

    # allocate
    mu    = fill(NaN, n+1, l)
    pi    = fill(NaN, n+1, l)
    muhat = fill(NaN, n+1, l)
    pihat = fill(NaN, n+1, l)
    da    = fill(NaN, n+1, l)
    v     = fill(NaN, n+1, l)
    w     = fill(NaN, n+1, l-1)

    # init
    mu[1,1]      = 1 / (1 + exp(-mu0[1]))
    pi[1,1]      = Inf
    mu[1,2:end]  = mu0[2:end]
    pi[1,2:end]  = 1 ./ sa0[2:end]

    for k in 2:(n+1)
        if !(k-1 in ign)
            # level 2 prediction
            muhat[k,2] = mu[k-1,2] 

            # level 1
            muhat[k,1] = 1/(1 + exp(-kappa[1]*muhat[k,2]))
            pihat[k,1] = 1/(muhat[k,1]*(1-muhat[k,1]))
            pi[k,1]    = Inf
            mu[k,1]    = obs2[k]
            da[k,1]    = mu[k,1] - muhat[k,1]

            # level 2 update
            pihat[k,2] = 1/(1/pi[k-1,2] + exp(kappa[2]*mu[k-1,3] + omega[2]))
            pi[k,2]    = pihat[k,2] + kappa[1]^2/pihat[k,1]
            mu[k,2]    = muhat[k,2] + kappa[1]/pi[k,2]*da[k,1]
            da[k,2]    = (1/pi[k,2] + (mu[k,2]-muhat[k,2])^2)*pihat[k,2] - 1

            # level 3 update
            j = l # set j = l = 3
            muhat[k,j] = mu[k-1,j] 
            pihat[k,j] = 1/(1/pi[k-1,j] + t2[k]*z_variance)
            # Weighting factor for the top-level volatility
            v[k,j]     = t2[k]*z_variance
            v[k,j-1]   = t2[k]*exp(kappa[j-1]*mu[k-1,j] + omega[j-1])
            w[k,j-1]   = v[k,j-1]*pihat[k,j-1]
            pi[k,j]    = pihat[k,j] + 0.5*kappa[j-1]^2*w[k,j-1]*
                         (w[k,j-1] + (2*w[k,j-1]-1)*da[k,j-1])
            mu[k,j]    = muhat[k,j] + 0.5*(1/pi[k,j])*kappa[j-1]*w[k,j-1]*da[k,j-1]
            da[k,j]    = (1/pi[k,j] + (mu[k,j]-muhat[k,j])^2)*pihat[k,j] - 1
        else
            # ignore trial → carry forward
            mu[k,:]    .= mu[k-1,:]
            pi[k,:]    .= pi[k-1,:]
            muhat[k,:] .= muhat[k-1,:]
            pihat[k,:] .= pihat[k-1,:]
            da[k,:]    .= da[k-1,:]
        end
    end

    # remove dummy trial
    mu    = mu[2:end,:]
    pi    = pi[2:end,:]
    muhat = muhat[2:end,:]
    pihat = pihat[2:end,:]
    da    = da[2:end,:]
    v     = v[2:end,:]
    w     = w[2:end,:]


    return mu, pi, muhat, pihat, da, v, w
end

"""
    simulate_responses_mathys(u, beta, mu0, sa0, kappa, omega, z_variance)
Perform a full simulate→respond step:
1. run HGF on `u`
2. compute choice probs with inverse temp beta
3. draw binary `0/1` responses

Returns `(y_sim, p)` where `y_sim` is Vector{Int} and `p` the choice probabilities.
"""
function simulate_responses_mathys(obs, beta, mu0, sa0, kappa, omega, z_variance)
    mu, pi, muhat, pihat, da, v, w = simulate_hgf_mathys(obs, mu0, sa0, kappa, omega, z_variance)
    sa = 1 ./ pi
    sahat = 1 ./ pihat
    prediction      = muhat[:,1]
    p      = prediction .^ beta ./ ((1 .- prediction) .^ beta .+ prediction .^ beta)
    resp  = rand.(Bernoulli.(p))
    let 
        p3 = plot(title = "Hidden State Z (third level)", legend = false)
        p2 = plot(title = "Hidden State X (second level)", legend = false)
        p1 = plot(title = "Posterior expectation of input (first level)", legend = :top, legendfontsize = 5)

        # Level 3
        plot!(p3, 1:n, mu[:,3], ribbon = sqrt.(sa[:,3]), color = :teal)
        
        # Level 2
        plot!(p2, 1:n, mu[:,2], ribbon = sqrt.(sa[:,2]), color = :green)
        
        # Level 1 - Posterior input
        posterior_input = 1 ./ (1 .+ exp.(-mu[:,2]))
        plot!(p1, 1:n, posterior_input, color = :violet, label = "Posterior input")

        # Observations
        scatter!(p1, 1:n, obs, color = :green, markershape = :circle, label = "Observation")

        # Shifted responses
        resp_shifted = [r == 1 ? 1.1 : (r == 0 ? -0.1 : NaN) for r in resp]
        scatter!(p1, 1:n, resp_shifted, color = :black, markershape = :xcross, label = "Response")

        # Show all plots with layout
        plot(p3, p2, p1, layout = @layout([ a; b; c ]))
    end




    return resp, p
end


x_initial_mean = 0.0
x_initial_variance = .10
z_initial_mean = 1.0
z_initial_variance = 1.0
kappa_param = 1
omega_param = -3
beta = .5
z_variance     = 0.0025
# rho is directional change in volatility, which we don't need here

mu0    = [NaN, x_initial_mean, z_initial_mean]
sa0    = [NaN, x_initial_variance, z_initial_variance]
kappa     = [kappa_param, kappa_param]
omega     = [NaN, omega_param]




obs = obs_data


y_sim, p_sim = simulate_responses_mathys(obs, beta, mu0, sa0, kappa, omega, z_variance)
println("Simulated responses: ", y_sim)
println("Choice probabilities: ", round.(p_sim, digits=3))






# Hierarchical Gaussian Filter: The probability of an observation is determined by the hidden state (x)  whose variance is determined by the hidden state (z).
# This probability is rounded to the nearest quarter (0.0, 0.25, 0.75, 1.0).
const QUARTERS = [0.0, 0.25, 0.75, 1.0]
# Helper function to snap a single value to the nearest quarter:
round_to_quarter(x::Real) = QUARTERS[argmin(abs.(QUARTERS .- x))]
# Simulate observations (obs), responses (resp), and hidden states (x,z) given parameters (z_precision, z_initial, x_initial, κ, ω, β).
function simulate_hgf_rxinfer(n, seed, z_precision, z_initial, x_initial, κ, ω, β)
    rng = StableRNG(seed)

    x_prev = x_initial 
    z_prev = z_initial 

    z = Vector{Float64}(undef, n)
    x = Vector{Float64}(undef, n)
    temp = Vector{Float64}(undef, n)
    obs = Vector{Float64}(undef, n)
    resp = Vector{Float64}(undef, n)

    for i in 1:n
        z[i] = rand(rng, Normal(z_prev, 1/sqrt(z_precision)))
        x[i] = rand(rng, Normal(x_prev, sqrt(exp((κ * z[i]) + ω))))
        obs[i] =  round_to_quarter(cdf(Normal(), x[i]))
        temp[i] = rand(Normal(dot(β, x[i]), 1))
        resp[i] = cdf(Normal(), temp[i])
        z_prev = z[i]
        x_prev = x[i]
    end

    return (z=z, x=x, obs=obs, temp=temp, resp=resp)

end

n = 100
seed = 23
z_precision = 1 # Precision of the hidden state z
z_initial = 0.0 # Initial value of the hidden state z
x_initial = 0.0 # Initial value of the hidden state x
κ = .001 # Coupling parameter linking state z to state x
ω = .001 # Coupling parameter linking state z to state x
β = 2  # Temperature parameter for the response

z, x, obs, temp, resp = simulate_hgf_rxinfer(n, seed, z_precision, z_initial, x_initial, κ, ω, β)


let 
    pz = plot(title = "Hidden States Z")
    px = plot(title = "Hidden States X")
    
    plot!(pz, 1:n, z, label = "z_i", color = :orange)
    plot!(px, 1:n, x, label = "x_i", color = :green)
    scatter!(px, 1:n, obs, label = "obs_i", color = :red, ms = 2, alpha = 0.2)
    scatter!(px, 1:n, resp, label = "resp_i", color = :blue, ms = 2, alpha = 0.2)
    
    plot(pz, px, layout = @layout([ a; b ]))
end