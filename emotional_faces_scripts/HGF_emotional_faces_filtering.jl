# Only run the setup code if this script is being run directly using the REPL; otherwise, it is being run with `include` in another script
run_setup_code = false
if run_setup_code
    # import Pkg
    # Pkg.add(["RxInfer", "DataFrames", "CSV", "Plots", "StatsPlots", "Distributions", 
    # "BenchmarkTools", "StableRNGs", "ExponentialFamilyProjection", "ExponentialFamily",
    # "ReactiveMP", "Cairo", "GraphPlot", "Random"])
    # Activate local environment, see `Project.toml`
    import Pkg; 

    # determine if running on cluster or locally

    println("Running locally...")
    Pkg.activate("L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/"); # Note that the Project and Manifest files are in the same directory as this script.
    # Click Julia: Activate this Environment to run the REPL
    subject_id = "AA003" # Use a prolific or local subject id e.g., 5a5ec79cacc75b00017aa095
    # Local subject IDs will always be 5 characters, while prolific IDs will always be 24
    if length(subject_id) == 5
        study = "local"
    elseif length(subject_id) == 24
        study = "prolific"
    else
        error("Invalid subject ID length. Must be 5 (local) or 24 (prolific).")
    end
    predictions_or_responses = "responses" # Haven't set up infrastructure to fit predictions
    results_dir = "L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_test"
    root = "L:/"
    # get current datetime
    using Dates
    now_time = now()
    formatted_time = "_" * Dates.format(now_time, "yyyy-mm-ddTHH_MM_SS")

    # These variables should get sent in from smoothing. Assign them here during debugging:
    kappa_mu = 1
    kappa_sigma = 1
    omega_mu = 1
    omega_sigma = 1
    beta_shape = 1
    beta_rate = 1
    x_mean_mu = 0
    x_mean_sigma = 1
    z_mean_mu = 0
    z_mean_sigma = 1

    
    Pkg.instantiate()  # Reinstall missing dependencies


    using RxInfer
    using ExponentialFamilyProjection, ExponentialFamily
    using Distributions
    using Plots, StatsPlots
    using StableRNGs
    using CSV, DataFrames

    # Import file containing callbacks functions and tagged logger
    include(root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/callbacks.jl")

    # Seed for reproducibility
    seed = 23
    rng = StableRNG(seed)

    # Read in the task data
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

    # For debugging, let's just use two observations and responses
    resp_data = resp_data[1:2]
    obs_data = obs_data[1:2]

    obs_data[2] = 0
    resp_data[2] = 0
    niterations = 3

    # Count the number of missing responses
    nan_responses = count(isnan, resp_data)
    println("Number of NaN Responses: ", nan_responses)

    # Exit if participant has 200 mising responses
    if nan_responses == 200
        println("Too many missing responses. Exiting.")
        exit()
    end


    # Replace NaN responses with the missing keyword
    resp_data = convert(Vector{Union{Missing, Float64}}, resp_data)
    resp_data .= ifelse.(isnan.(resp_data), missing, resp_data)

    # If there are missing responses, we won't be able to compute Bethe Free Energy values because the graph is incomplete
    # The RxInfer developers seem to be working on this issue, so we can check later if it's been resolved
    if nan_responses > 0
        compute_bethe_free_energy = false
    else
        compute_bethe_free_energy = true
    end

else
    # assign parameter values from the results of the smoothing script
    kappa_mu = mean(κ_posterior)
    kappa_sigma = std(κ_posterior)
    omega_mu = mean(ω_posterior)
    omega_sigma = std(ω_posterior)
    beta_shape = shape(β_posterior)
    beta_rate = 1/scale(β_posterior)
    x_mean_mu = mean(x_initial_posterior)
    x_mean_sigma = std(x_initial_posterior)
    z_mean_mu = mean(z_initial_posterior)
    z_mean_sigma = std(z_initial_posterior)
end




# todo: esitmate variance on top level (z_variance). could potentially fix the middle mean to .5; 
@model function hgf_filtering(obs, resp, z_mean_mu, z_mean_sigma, x_mean_mu, x_mean_sigma, kappa_mu, kappa_sigma, omega_mu, omega_sigma, beta_shape, beta_rate)
    # Initial states - adjust means to be closer to data range
    # z_variance ~ Gamma(shape = 1.0, rate = 1.0)  # Reduced variance
    z_mean ~ Normal(mean = z_mean_mu, variance = z_mean_sigma)  where { pipeline = TaggedLogger("z_mean") }# Reduced variance
    x_mean ~ Normal(mean = x_mean_mu, variance = x_mean_sigma)  where { pipeline = TaggedLogger("x_mean") }# Reduced variance
    

    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = kappa_mu, variance = kappa_sigma)  where { pipeline = TaggedLogger("κ") }# Reduced variance and mean
    ω ~ Normal(mean = omega_mu, variance = omega_sigma)  where { pipeline = TaggedLogger("ω") }# Reduced variance
    β ~ Gamma(shape = beta_shape, rate = beta_rate)  where { pipeline = TaggedLogger("β") } # Less constrained


    # Higher layer update (Gaussian random walk)
    z ~ Normal(mean = z_mean, variance = 1)  where { pipeline = TaggedLogger("z") }

    # Lower layer update
    x ~ GCV(x_mean, z, κ, ω) where { pipeline = TaggedLogger("x") }

    obs ~ Probit(x) where { pipeline = TaggedLogger("obs") }

    temp ~ softdot(β, x, 1.0) where { pipeline = TaggedLogger("temp") }
    resp ~ Probit(temp) where { pipeline = TaggedLogger("resp") }
     
end

 # Visualize model
model_generator = hgf_filtering() | (obs = [1], resp = [1], kappa_mu=1,kappa_sigma=1,omega_mu=1,omega_sigma=1,beta_shape=1,beta_rate=1,z_mean_mu=1,z_mean_sigma=1,x_mean_mu=1,x_mean_sigma=1,)
model_to_plot   = RxInfer.getmodel(RxInfer.create_model(model_generator))
using GraphViz
# Call `load` function from `GraphViz` to visualise the structure of the graph
GraphViz.load(model_to_plot, strategy = :simple)
 
@constraints function hgfconstraints_filtering() 
    #Structured mean-field factorization constraints
    q(x, temp, z, κ, ω, β, x_mean) = q(x, temp, x_mean)q(z)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
   # q(z_variance) :: ProjectedTo(Gamma)    
end

autoupdates = @autoupdates begin 
    z_mean_mu, z_mean_sigma = mean_var(q(z))
    x_mean_mu, x_mean_sigma = mean_var(q(x))
end

import ReactiveMP: as_companion_matrix, ar_transition, getvform, getorder, add_transition!
# Add this method to prevent compiler ambiguity for the product of colliding messages
Base.prod(::GenericProd, left::ProductOf{L, R}, ::Missing) where {L, R} = left


@meta function hgfmeta_filtering()
    # Lets use 31 approximation points in the Gauss Hermite cubature approximation method
    GCV() -> GCVMetadata(GaussHermiteCubature(31))
end



function run_inference_filtering(obs, resp, niterations)
    @initialization function hgf_init_filtering()
        q(x) = NormalMeanVariance(x_mean_mu, x_mean_sigma)
        q(z) = NormalMeanVariance(z_mean_mu, z_mean_sigma)
        q(κ) = NormalMeanVariance(kappa_mu, kappa_sigma)
        q(ω) = NormalMeanVariance(omega_mu, omega_sigma)
        q(β) = GammaShapeRate(beta_shape, beta_rate) # i have a feeling these won't matter too much; will want to be consistent with smoothing
    end

    return infer(
        model = hgf_filtering(kappa_mu=kappa_mu, kappa_sigma=kappa_sigma, omega_mu=omega_mu, omega_sigma=omega_sigma, beta_shape=beta_shape, beta_rate=beta_rate,),  # Reduced z_variance
        data = (obs = obs, resp = resp,),
        meta = hgfmeta_filtering(),
        constraints = hgfconstraints_filtering(),
        initialization = hgf_init_filtering(),
        iterations = niterations,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (:x, :z, :ω, :κ, :β, :temp, :z_mean, :x_mean),
        free_energy = compute_bethe_free_energy,  # Compute Bethe Free Energy when there are no missing values
        free_energy_diagnostics = nothing, # turns off the error for NaN or inf FE
        showprogress = true,
        keephistory   = 1000,
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
        ),
        autoupdates = autoupdates,
        addons = (AddonMemory(),), # uncomment this to get a more detailed description of the message computation
        autostart = true, # might want this to be false?
    )
end



# For ease of debugging, output to a file instead of the console
if get(ENV, "CLUSTER", "false") == "true"
    infer_result = run_inference_filtering(obs_data, resp_data, niterations)
else
    # Open a log file with w, meaning it gets overwritten each time
    logfile = open(results_dir*"/inference_filtering.log", "w")
    # send all subsequent prints/errors into logfile…
    redirect_stdout(logfile) do
    redirect_stderr(logfile) do
        infer_result = run_inference_filtering(obs_data, resp_data, niterations)
    end
    end
    close(logfile)
    # also run it here so you have access to the result
    infer_result = run_inference_filtering(obs_data, resp_data, niterations)
end

# Extract the marginal distributions for the hidden states
x_mean_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:x_mean]]
z_mean_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:z_mean]]
x_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:x]]
z_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:z]]
ω_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:ω]]
κ_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:κ]]
β_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:β]]
temp_marginals = [getfield.(marginals, :data) for marginals in infer_result.history[:temp]]

# List of marginals to process, replace with your actual data
all_marginals = Dict(
    "x_mean" => x_mean_marginals,
    "z_mean" => z_mean_marginals,
    "x"      => x_marginals,
    "z"      => z_marginals,
    "ω"      => ω_marginals,
    "κ"      => κ_marginals,
    "β"      => β_marginals,
    "temp"   => temp_marginals
)

# Collect records as NamedTuples for the DataFrame
rows = NamedTuple[]

for (param_name, marginals_list) in all_marginals
    for (trial_idx, trial) in enumerate(marginals_list)
        for (iter_idx, dist) in enumerate(trial)
            if dist isa NormalMeanVariance
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "mean",
                             value = dist.μ))
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "variance",
                             value = dist.v))
            elseif dist isa NormalWeightedMeanPrecision
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "weighted_mean",
                             value = dist.xi))
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "precision",
                             value = dist.w))
            elseif dist isa Gamma
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "shape",
                             value = dist.α))
                push!(rows, (parameter = param_name,
                             trial = trial_idx,
                             iteration = iter_idx,
                             stat_type = "scale",
                             value = dist.θ)) 
            else
                @warn "Unknown distribution type for $param_name at trial $trial_idx iteration $iter_idx"
            end
        end
    end
end

marginal_posteriors = DataFrame(rows)

