
# print current working directory
println("Current working directory: ", pwd())
     # install all deps listed in Manifest.toml
using RxInfer                     # Bayesian DSL for Julia
using RxInfer: @node           # import the custom node macro
using Distributions               # For base distributions
using Random             # for logpdf and insupport

function ddm(v::Float64, a::Float64, w::Float64, T::Float64)
    # Discrete Euler approximation of diffusion to threshold crossing
    dt = 0.01
    x = w * a         # starting point
    t = 0.0           # accumulator for time
    choice = 0       # placeholder

    while true
        # sample incremental evidence
        inc = rand(Normal(v * dt, sqrt(dt)))
        x += inc
        t += dt
        if x >= a
            choice = 1
            break
        elseif x <= 0
            choice = 0
            break
        end
    end

    # add non-decision time
    rt = t + T
    return choice, rt
end

# --- Direct copy from SequentialSamplingModels.jl/src/DDM.jl ---
mutable struct DDM{T <: Real}
    ν::T
    α::T
    z::T
    τ::T
end
function DDM(ν, α, z, τ)
    return DDM(promote(ν, α, z, τ)...)
end
function params(d::DDM)
    return (d.ν, d.α, d.z, d.τ)
end

# PDF / logPDF machinery
function pdf(d::DDM, choice, rt; ϵ::Real = 1e-12)
    if choice == 1
        (ν, α, z, τ) = params(d)
        return _pdf(DDM(-ν, α, 1 - z, τ), rt; ϵ)
    end
    return _pdf(d, rt; ϵ)
end

function _pdf(d::DDM{T}, t::Real; ϵ::Real = 1e-12) where {T <: Real}
    (ν, α, z, τ) = params(d)
    if τ >= t
        return zero(T)
    end
    u = (t - τ) / α^2
    # term counts
    K_s = 2.0
    if (2 * sqrt(2π * u) * ϵ) < 1
        K_s = max(2 + sqrt(-2u * log(2ϵ * sqrt(2π * u))), sqrt(u) + 1)
    end
    K_l = 1 / (π * sqrt(u))
    if (π * u * ϵ) < 1
        K_l = max(sqrt((-2 * log(π * u * ϵ)) / (π^2 * u)), K_l)
    end
    p = exp((-α * z * ν) - (0.5 * ν^2 * (t - τ))) / α^2
    if K_s < K_l
        return p * _small_time_pdf(u, z, ceil(Int, K_s))
    else
        return p * _large_time_pdf(u, z, ceil(Int, K_l))
    end
end

function _small_time_pdf(u::T, z::T, K::Int) where {T <: Real}
    inf_sum = zero(T)
    series = (-floor(Int, 0.5*(K-1))):ceil(Int, 0.5*(K-1))
    for k in series
        inf_sum += ((2k + z) * exp(-((2k + z)^2)/(2u)))
    end
    return inf_sum / sqrt(2π * u^3)
end

function _large_time_pdf(u::T, z::T, K::Int) where {T <: Real}
    inf_sum = zero(T)
    for k = 1:K
        inf_sum += (k * exp(-0.5 * (k^2 * π^2 * u)) * sin(k * π * z))
    end
    return π * inf_sum
end

function P_upper(ν::T, α::T, z::T) where {T <: Real}
    e = exp(-2 * ν * α * (1 - z))
    if isinf(e)
        return one(T)
    elseif abs(e - 1) < sqrt(eps(T))
        return one(T) - z
    end
    return (1 - e) / (exp(2 * ν * α * z) - e)
end

function _exp_pnorm(a::T, b::T) where {T <: Real}
    r = exp(a) * cdf(Normal(), b)
    if isnan(r) && b < -5.5
        r = (1/sqrt(2)) * exp(a - b^2/2) * (0.5641882/(b^3) - 1/(b*sqrt(π)))
    end
    return r
end

logpdf(d::DDM, choice, rt; ϵ::Real = 1e-12) = log(pdf(d, choice, rt; ϵ))

# Support check for DDM
BayesBase.insupport(d::DDM, xy::Tuple{Int,Float64}) = (xy[1] in (0,1)) && (xy[2] > d.τ)

@model function rl_ddm_model(tone, intensity, observed)
    # --- Priors ---
    α ~ Beta(1,1)                              # learning rate
    β ~ Gamma(2,1)                             # inverse temperature
    V0 ~ Beta(1,1)                             # initial Q
    η ~ Beta(1,1)                              # forgetting rate

    T ~ truncated(Normal(0.3,0.1), 0.0, Inf)   # non-decision time
    w ~ Beta(1,1)                              # starting bias
    a_base ~ truncated(Normal(1.0,0.5), 0.0, Inf) # base threshold

    drift_rate_scalar_low  ~ Normal(0,1)       # mapping-specific scalers
    drift_rate_scalar_high ~ Normal(0,1)
    boundary_scalar_sad   ~ Normal(0,1)
    boundary_scalar_angry ~ Normal(0,1)

    # Initialize belief
    q = V0

    for t in eachindex(tone)
        # Compute trial-wise drift
        v_scalar = observed[t]==1 ? drift_rate_scalar_high : drift_rate_scalar_low
        v_t = v_scalar * intensity[t]

        # Compute trial-wise boundary
        q_diff = 2q - 1
        bs = observed[t]==1 ? boundary_scalar_sad : boundary_scalar_angry
        a_t = a_base * (1 + bs * q_diff)

        # Use nested DDM model for simulation
        (choice_ddm[t], rt[t]) = ddm(v_t, a_t, w, T)

        # RL choice (belief about mapping)
        p_choice = exp(β*q)/(exp(β*q)+exp(β*(1-q)))
        choice_rl[t] ~ Bernoulli(p_choice)

        # Reward (correct belief?)
        r = choice_rl[t]==observed[t] ? 1.0 : 0.0

        # Q-value update & forgetting
        q = q + α*(r - q)
        q = (1-η)*q + η*V0
    end
end

# Example inputs (200 trials)
tone_data     = rand(Bernoulli(0.5), 200)
intensity_data= rand([0.0,0.25,0.75,1.0], 200)
observed_data = rand(Bernoulli(0.5), 200)


# Manual simulation of joint RL-DDM synthetic data
# Set scalar parameter values (example values, set as needed)
α_val = 0.2
β_val = 2.0
V0_val = 0.5
η_val = 0.1
T_val = 0.3
w_val = 0.5
a_base_val = 1.0
drift_rate_scalar_low_val = -0.5
drift_rate_scalar_high_val = 0.5

boundary_scalar_sad_val = 0.3
boundary_scalar_angry_val = -0.2

n_trials = length(tone_data)
choice_sim = Vector{Int}(undef, n_trials)
rt_sim     = Vector{Float64}(undef, n_trials)
q = V0_val  # set your initial Q0 (e.g., V0)
for t in 1:n_trials
    # compute drift and boundary as in model
    v_scalar = observed_data[t]==1 ? drift_rate_scalar_high_val : drift_rate_scalar_low_val
    v_t = v_scalar * intensity_data[t]
    q_diff = 2 * q - 1
    bs = observed_data[t]==1 ? boundary_scalar_sad_val : boundary_scalar_angry_val
    a_t = a_base_val * (1 + bs * q_diff)

    # simulate DDM
    choice_sim[t], rt_sim[t] = ddm(v_t, a_t, w_val, T_val)

    # simulate RL choice and update belief
    p_choice = exp(β_val*q) / (exp(β_val*q) + exp(β_val*(1-q)))
    choice_rl_t = rand(Bernoulli(p_choice))
    r = (choice_rl_t == observed_data[t]) ? 1.0 : 0.0
    q = q + α_val*(r - q)
    q = (1-η_val)*q + η_val*V0_val
end