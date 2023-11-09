using Tables

import CounterfactualExplanations.DataPreprocessing.CounterfactualData

"""
Generate a Python dataframe using values from x and given column names.
"""
function to_python_df(x::AbstractArray, colnames::Vector{String})
    PythonDataFrame(columns=colnames, data=x)
end

to_python_df(x::AbstractArray, model::SklearnModel) = to_python_df(x, [s for s ∈ model.X.columns])

"""
Convert Python dataframe to Tables.MatrixTable
"""
function to_matrix_table(X)
    Tables.MatrixTable([Symbol(x) for x in X.columns],
                       Dict(Symbol(key) => index
                            for (index, key) ∈ enumerate(X.columns)),
                       X.values)
end

"""
Generate a CounterfactualData from a SklearnModel, using its training data.
"""
function CounterfactualData(model::SklearnModel)
    index_for_feature = Dict(value => index
                             for (index, value) ∈ enumerate(model.X.columns))
    cat_indices = [[index_for_feature[feat]]
                   for feat in model.categorical ]
    # an empty list will otherwise crash the constructor
    if isempty(cat_indices)
        cat_indices = nothing
    end
    num_indices = [ index_for_feature[feat] for feat in model.continuous ]
    if isempty(num_indices)
        num_indices = nothing
    end
    formatted_X = to_matrix_table(model.X)
    y_as_matrix = reshape(model.y.values, (length(model.y.values), 1))
    CounterfactualData(formatted_X,
                       model.y.values,
                       features_categorical=cat_indices,
                       features_continuous=num_indices)
end

const ALLOWED_CLASSIFIERS = Set([
    "RandomForestClassifier"
])

function get_categorical(data::CounterfactualData)
    if isnothing(data.features_categorical)
        []
    else
        data.features_categorical
    end
end

function get_continuous(data::CounterfactualData)
    if isnothing(data.features_continuous)
        []
    else
        data.features_continuous
    end
end

function one_hot_decode_to_python_series(y::AbstractArray)
    print([argmax(y[:,col])-1 for col in 1:size(y,2)])
    PythonSeries(data=[argmax(y[:,col])-1 for col in 1:size(y,2)],
                 name="target")
end

"""
Generate a SklearnModel for given prepared data, using the specified classifier.
"""
function SklearnModel(data::CounterfactualData,
                      colnames::Vector{String},
                      classifier::String)
    if classifier ∉ ALLOWED_CLASSIFIERS
        throw(ArgumentError("Unknown classifier: $classifier"))
    end
    classifiers_as_string = join(ALLOWED_CLASSIFIERS, ",")
    py_model = py"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import $$classifiers_as_string
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def gen_model(X, y, categorical, numerical, classifier):
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    transformations = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical),
            ("cat", categorical_transformer, categorical),
        ]
    )
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(
        steps=[("preprocessor", transformations), ("classifier", classifier)]
    )
    model = clf.fit(X, y)
    return model
"""
    caller = py"gen_model"
    # format data for scikit-learn
    X = to_python_df(data.X', colnames)
    y = one_hot_decode_to_python_series(data.y)
    likelihood = if size(data.y, 1) > 2
        :classification_multi
    else
        :classification_binary
    end
    categorical = [colnames[x[0]] for x in get_categorical(data)]
    numerical = [colnames[x] for x in get_continuous(data)]
    classifier = py"$$classifier()"
    py_model = caller(X, y, categorical, numerical, classifier)
    SklearnModel(py_model,
                 X,
                 y,
                 categorical,
                 numerical,
                 likelihood)
end
