module WienerFactor
#─────────────────────────────────────────────────────────────────────────────
#  Drift–Diffusion “Wiener” factor for RxInfer / GraphPPL
#  • Uses Navarro & Fuss (2009) cosine-series to evaluate the log-pdf
#  • One output interface  :out     (the observed trial)
#  • Four parent interfaces :a, :v, :z, :t0   (latent DDM parameters)
#  • Provides the three hooks RxInfer needs:  logpdf, parents, interfaces
#─────────────────────────────────────────────────────────────────────────────

using ReactiveMP
import ReactiveMP: logpdf, parents, interfaces       # extension points
import RxInfer: ReactiveMPGraphPPLBackend            # backend for 2-arg interfaces
using Distributions
using ForwardDiff         # handy if you later want gradients

export Wiener, wiener_logpdf

#─────────────────────────────────────────────────────────────────────────────
#  Factor type
#─────────────────────────────────────────────────────────────────────────────

"""
    Wiener(; a, v, z, t0)

A deterministic factor that turns latent DDM parameters
`(a, v, z, t0)` into the likelihood for one reaction-time / choice pair.
"""
struct Wiener{T}
    params :: T   # NamedTuple(:a,:v,:z,:t0)
end
Wiener(; a, v, z, t0) = Wiener((a = a, v = v, z = z, t0 = t0))

#─────────────────────────────────────────────────────────────────────────────
#  First-passage log-density  (Navarro & Fuss, 2009)
#─────────────────────────────────────────────────────────────────────────────

"""
    wiener_logpdf(rt, c; a, v, z, t0, s = 1.0, k = 20)

Return `log p(rt, c | a,v,z,t0)` for a DDM with diffusion `s`.
`c` is the choice: 0 = upper/right boundary, 1 = lower/left.
`k` is the truncation level of the cosine series (≥20 ≈ 1 ms precision).
"""
function wiener_logpdf(rt, c; a, v, z, t0, s = 1.0, k = 20)
    rt ≤ t0 && return -Inf
    t   = rt - t0
    φ   = v / s^2
    x₀  = c == 0 ? z * a : (1 - z) * a               # dist. to chosen bdry
    pref = (π / a^2) * exp(-φ*x₀ - φ^2*s^2*t/2) / s^2

    acc = 0.0
    for n in 1:k
        λ = n * π / a
        acc += λ * exp(-λ^2 * s^2 * t / 2) * sin(λ * x₀)
    end
    pdf = pref * acc
    return log(pdf + eps())        # small eps to avoid log(0)
end

#─────────────────────────────────────────────────────────────────────────────
#  RxInfer / ReactiveMP hooks
#─────────────────────────────────────────────────────────────────────────────

# 1.  logpdf  — called when the engine evaluates the factor
function logpdf(w::Wiener, data)
    rt = getfield(data, :rt)
    c  = getfield(data, :c)
    p  = w.params
    return wiener_logpdf(rt, c; a = p.a, v = p.v, z = p.z, t0 = p.t0)
end

# 2.  parents — list of latent inputs
parents(::Type{<:Wiener})    = (:a, :v, :z, :t0)

# 3.  interfaces
# – plain version used by GraphPPL
interfaces(::Type{<:Wiener}) = (:out, :a, :v, :z, :t0)
# – backend-specific overload used by newer RxInfer versions
interfaces(::ReactiveMPGraphPPLBackend, ::Type{<:Wiener}, ::Any) =
    (:out, :a, :v, :z, :t0)

end # module