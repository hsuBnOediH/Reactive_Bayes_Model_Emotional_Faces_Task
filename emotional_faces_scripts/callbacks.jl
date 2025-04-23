function before_model_creation()
    println("The model is about to be created")
end

function after_model_creation(model::ProbabilisticModel)
    println("The model has been created")
    println("  The number of factor nodes is: ", length(RxInfer.getfactornodes(model)))
    println("  The number of latent states is: ", length(RxInfer.getrandomvars(model)))
    println("  The number of data points is: ", length(RxInfer.getdatavars(model)))
    println("  The number of constants is: ", length(RxInfer.getconstantvars(model)))
end

function before_inference(model::ProbabilisticModel)
    println("The inference procedure is about to start")
end

function after_inference(model::ProbabilisticModel)
    println("The inference procedure has ended")
end

function before_iteration(model::ProbabilisticModel, iteration::Int)
    println("The iteration ", iteration, " is about to start")
end

function after_iteration(model::ProbabilisticModel, iteration::Int)
    println("The iteration ", iteration, " has ended")
end

function before_data_update(model::ProbabilisticModel, data)
    println("The data is about to be processed")
end

function after_data_update(model::ProbabilisticModel, data)
    println("The data has been processed")
end

function on_marginal_update(model::ProbabilisticModel, name, update)
    # Check if the name is :x or :ω (Symbols)
    # if name == :κ || name == :ω
    #     println("New marginal update for ", name, ": ", update)
    # end
    println("New marginal update for ", name, ": ", update,"\n")

end


# Define the LoggerPipelineStage to print out messages
# 1) single‐Val interface
import ReactiveMP: AbstractPipelineStage, apply_pipeline_stage, as_message, functionalform

# 1) Define a new pipeline‐stage type that carries a String label
struct TaggedLogger <: AbstractPipelineStage
  name::String
end

# 2) Override apply_pipeline_stage for non‑indexed ports
function apply_pipeline_stage(
    stage::TaggedLogger,
    factornode,
    tag::Val{T},
    stream
) where {T}
  return stream |> tap(v -> begin
    real = as_message(v)
    println("[$(stage.name)] [$(functionalform(factornode)) / $T] → $real")
  end)
end

# 3) And for indexed ports
function apply_pipeline_stage(
    stage::TaggedLogger,
    factornode,
    tag::Tuple{Val{T},Int},
    stream
) where {T}
  return stream |> tap(v -> begin
    real = as_message(v)
    println("[$(stage.name)] [$(functionalform(factornode)) / $T:$(tag[2])] → $real")
  end)
end
