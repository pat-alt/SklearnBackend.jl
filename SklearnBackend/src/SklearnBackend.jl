__precompile__()

module SklearnBackend

using CounterfactualExplanations
using PyCall
using Conda
Conda.add("scikit-learn")
Conda.add("pandas")

# Used to construct Python dataframes
# Due to how PyCall works, cannot be precompiled, therefore has to be set in
# __init__()
const PythonDataFrame = PyNULL()
const PythonSeries = PyNULL()

function __init__()
    copy!(PythonDataFrame, pyimport_conda("pandas", "").DataFrame)
    copy!(PythonSeries, pyimport_conda("pandas", "").Series)
end

struct SklearnModel <: AbstractDifferentiableModel
    model::PyCall.PyObject # The fitted model Python object
    X::PyCall.PyObject     # Training data used for fitting (Pandas dataframe)
    y::PyCall.PyObject     # Outcome of training data
    categorical::PyCall.PyObject  # names of columns containing categorical data
    continuous::PyCall.PyObject
    likelihood::Symbol
end

# import required functions for custom CounterfactualExplanations model
import CounterfactualExplanations.Models: logits
import CounterfactualExplanations.Models: probs 

function probs(model::SklearnModel, input::AbstractArray)
    X = permutedims(input)
    [t for t in transpose(model.model.predict_proba(to_python_df(X, model)))]
end
# Some scikit-learn classifiers do not rely on logits, therefore we cannot
# assume that they are generally available. We use probabilities.
logits(model::SklearnModel, input::AbstractArray) = probs(model, input)

"""
Select the first encountered factual, i.e. the first entry classified as `outcome`.
"""
function select_first_factual(data::CounterfactualData, outcome::Int)
    for entry in 1:size(data.X, 2)
        if data.y[outcome,entry] > 0
            return reshape(data.X[:,entry], (size(data.X, 1), 1))
        end
    end
    throw(ArgumentError("No factual found for outcome ", outcome))
end

include("sample_model.jl")
include("data_conversion.jl")

export gen_sample_model, CounterfactualData, SklearnModel, select_first_factual

end # module SklearnBackend
