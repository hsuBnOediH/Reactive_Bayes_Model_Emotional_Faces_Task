# Only run the setup code if this script is being run directly
# Otherwise, just important the functions and run the model
if abspath(PROGRAM_FILE) == @__FILE__


    # import Pkg
    # Pkg.add(["RxInfer", "DataFrames", "CSV", "Plots", "StatsPlots", "Distributions", 
    # "BenchmarkTools", "StableRNGs", "ExponentialFamilyProjection", "ExponentialFamily",
    # "ReactiveMP", "Cairo", "GraphPlot", "Random"])
    # Activate local environment, see `Project.toml`
    import Pkg; 

    # determine if running on cluster or locally
    if get(ENV, "CLUSTER", "false") == "true"
        println("Running on the cluster...")
        # println(ENV)
        println("SUBJECT: ", get(ENV, "SUBJECT", "NOT SET"))
        println("PREDICTIONS_OR_RESPONSES: ", get(ENV, "PREDICTIONS_OR_RESPONSES", "NOT SET"))
        println("RESULTS: ", get(ENV, "RESULTS", "NOT SET"))

        Pkg.activate("/media/labs/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/cluster_environment/")
        subject_id = get(ENV, "SUBJECT", "SUBJECT_NOT_SET")
        predictions_or_responses = get(ENV, "PREDICTIONS_OR_RESPONSES","PREDICTIONS_OR_RESPONSES_NOT_SET")
        results_dir = get(ENV, "RESULTS","RESULTS_NOT_SET")
        root = "/media/labs/"
        formatted_time = ""
        hyperparam_str = get(ENV, "HYPERPARAM_STR", "NOT SET")
        println("Hyperparams String: ", hyperparam_str)
        hyperparam_dict = Dict(
            pair[1] => parse(Float64, pair[2]) for pair in 
            (split(kv, "=") for kv in split(hyperparam_str, ","))
        )
        prior_kappa_mean = get(hyperparam_dict, "prior_kappa_mean", "NOT SET")
        prior_kappa_variance = get(hyperparam_dict, "prior_kappa_variance", "NOT SET")
        prior_omega_mean = get(hyperparam_dict, "prior_omega_mean", "NOT SET")
        prior_omega_variance = get(hyperparam_dict, "prior_omega_variance", "NOT SET")
        prior_beta_shape = get(hyperparam_dict, "prior_beta_shape", "NOT SET")
        prior_beta_rate = get(hyperparam_dict, "prior_beta_rate", "NOT SET")
        prior_z_prev_mean = get(hyperparam_dict, "prior_z_prev_mean", "NOT SET")
        prior_z_prev_variance = get(hyperparam_dict, "prior_z_prev_variance", "NOT SET")
        prior_x_prev_mean = get(hyperparam_dict, "prior_x_prev_mean", "NOT SET")
        prior_x_prev_variance = get(hyperparam_dict, "prior_x_prev_variance", "NOT SET")
        initial_z_mean = get(hyperparam_dict, "initial_z_mean", "NOT SET")
        initial_z_variance = get(hyperparam_dict, "initial_z_variance", "NOT SET")
        initial_kappa_mean = get(hyperparam_dict, "initial_kappa_mean", "NOT SET")
        initial_kappa_variance = get(hyperparam_dict, "initial_kappa_variance", "NOT SET")
        initial_omega_mean = get(hyperparam_dict, "initial_omega_mean", "NOT SET")
        initial_omega_variance = get(hyperparam_dict, "initial_omega_variance", "NOT SET")
        initial_beta_shape = get(hyperparam_dict, "initial_beta_shape", "NOT SET")
        initial_beta_rate = get(hyperparam_dict, "initial_beta_rate", "NOT SET")
        niterations = get(hyperparam_dict, "niterations", "NOT SET")
        niterations = Int(niterations) # cast as an integer
    else
        println("Running locally...")
        Pkg.activate("L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_scripts/"); # Note that the Project and Manifest files are in the same directory as this script.
        # Click Julia: Activate this Environment to run the REPL
        subject_id = "5a5ec79cacc75b00017aa095"
        predictions_or_responses = "responses" # Haven't set up infrastructure to fit predictions
        results_dir = "L:/rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/model_output_prolific/hgf_RxInfer_test"
        root = "L:/"
        # get current datetime
        using Dates
        now_time = now()
        formatted_time = "_" * Dates.format(now_time, "yyyy-mm-ddTHH_MM_SS")

        prior_kappa_mean = 1.0
        prior_kappa_variance = 0.1
        prior_omega_mean = 0.0
        prior_omega_variance = .01 # used to be .01
        prior_beta_shape = 1.0
        prior_beta_rate = 1.0
        prior_z_prev_mean = 0.0
        prior_z_prev_variance = 1.0
        prior_x_prev_mean = 0.0
        prior_x_prev_variance = 1.0
        initial_z_mean = 0.0
        initial_z_variance = 1.0
        initial_kappa_mean = 1.0
        initial_kappa_variance = 0.1
        initial_omega_mean = 0.0
        initial_omega_variance = 0.01
        initial_beta_shape = 0.1
        initial_beta_rate = 0.1
        niterations = 20

        posterior_beta_shape = 1.555
        posterior_beta_rate =1.555
    end
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
    file_name = root * "rsmith/lab-members/cgoldman/Wellbeing/emotional_faces/RxInfer_scripts/emotional_faces_processed_data/task_data_$(subject_id)_$(predictions_or_responses).csv"
    data = CSV.read(file_name, DataFrame)
    # Create variable for observations
    obs_data = data.observed 
    obs_data = Float64.(obs_data)
    # Create variable for responses
    resp_data = data.response
    resp_data = Float64.(resp_data)

    # For debugging, let's just use two observations and responses
    resp_data = resp_data[1:2]
    obs_data = obs_data[1:2]

    obs_data[2] = 0
    resp_data[2] = 0
    niterations = 1

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
end




# todo: esitmate variance on top level (z_variance). could potentially fix the middle mean to .5; 
@model function hgf_filtering(obs, resp, prior_z_prev_mean, prior_z_prev_variance, prior_x_prev_mean, prior_x_prev_variance, prior_kappa_mean, prior_kappa_variance, prior_omega_mean, prior_omega_variance, prior_beta_shape, prior_beta_rate)
    # Initial states - adjust means to be closer to data range
    # z_variance ~ Gamma(shape = 1.0, rate = 1.0)  # Reduced variance
    z_initial ~ Normal(mean = prior_z_prev_mean, variance = prior_z_prev_variance)  where { pipeline = TaggedLogger("z_initial") }# Reduced variance
    x_initial ~ Normal(mean = prior_x_prev_mean, variance = prior_x_prev_variance)  where { pipeline = TaggedLogger("x_initial") }# Reduced variance
    

    # Priors on κ and ω - maybe adjust these
    κ ~ Normal(mean = prior_kappa_mean, variance = prior_kappa_variance)  where { pipeline = TaggedLogger("κ") }# Reduced variance and mean
    ω ~ Normal(mean = prior_omega_mean, variance = prior_omega_variance)  where { pipeline = TaggedLogger("ω") }# Reduced variance
    β ~ Gamma(shape = prior_beta_shape, rate = prior_beta_rate)  where { pipeline = TaggedLogger("β") } # Less constrained


    # Higher layer update (Gaussian random walk)
    z ~ Normal(mean = z_initial, variance = 1)  where { pipeline = TaggedLogger("z") }

    # Lower layer update
    x ~ GCV(x_initial, z, κ, ω) where { pipeline = TaggedLogger("x") }

    # Noisy binary observations (Bernoulli likelihood)
    obs ~ Probit(x) where { pipeline = TaggedLogger("obs") }

    # Noisy binary response (Bernoulli likelihood)
    temp ~ softdot(β, x, 1.0) where { pipeline = TaggedLogger("temp") }
    resp ~ Probit(temp) where { pipeline = TaggedLogger("resp") }
     
end

 # Visualize model
model_generator = hgf_filtering() | (obs = [1], resp = [1],)
model_to_plot   = RxInfer.getmodel(RxInfer.create_model(model_generator))
using GraphViz
# Call `load` function from `GraphViz` to visualise the structure of the graph
GraphViz.load(model_to_plot, strategy = :simple)
 
@constraints function hgfconstraints_filtering() 
    #Structured mean-field factorization constraints
    #q(x, temp, z, κ, ω, β, x_initial,z_variance) = q(x, temp, x_initial)q(z,z_variance)q(κ)q(ω)q(β)
    q(x, temp, z, κ, ω, β, x_initial) = q(x, temp, x_initial)q(z)q(κ)q(ω)q(β)
    q(β) :: ProjectedTo(Gamma)
   # q(z_variance) :: ProjectedTo(Gamma)    
end

autoupdates = @autoupdates begin 
    prior_z_prev_mean, prior_z_prev_variance = mean_var(q(z))
    prior_x_prev_mean, prior_x_prev_variance = mean_var(q(x))
    # prior_kappa_mean, prior_kappa_variance = mean_var(q(κ)) 
    # prior_omega_mean, prior_omega_variance = mean_var(q(ω))
    # prior_beta_shape = shape(q(β))
    # prior_beta_rate = rate(q(β))
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
        q(x) = NormalMeanVariance(0, .06)
        q(z) = NormalMeanVariance(0, .02)
        q(κ) = NormalMeanVariance(0, .03)
        q(ω) = NormalMeanVariance(0, .04)
        q(β) = GammaShapeRate(.05, .05)
    end

    return infer(
        model = hgf_filtering(prior_kappa_mean=1,prior_kappa_variance=1,prior_omega_mean=1,prior_omega_variance=1,prior_beta_shape=1,prior_beta_rate=1,),  # Reduced z_variance
        data = (obs = obs, resp = resp,),
        meta = hgfmeta_filtering(),
        constraints = hgfconstraints_filtering(),
        initialization = hgf_init_filtering(),
        iterations = 10,  # More iterations
        options = (limit_stack_depth = 500,),  # Increased stack depth
        returnvars = (:x, :z, :ω, :κ, :β, :temp),
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
    logfile = open(results_dir*"/inference.log", "w")
    # send all subsequent prints/errors into logfile…
    redirect_stdout(logfile) do
    redirect_stderr(logfile) do
        infer_result = run_inference_filtering(obs_data, resp_data, niterations)
        # you can still use infer_result here if you like…
    end
    end
    close(logfile)
end