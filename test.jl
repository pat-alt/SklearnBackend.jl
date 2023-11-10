using Pkg; Pkg.activate("SklearnBackend/")
using OhMyREPL
using Revise

using CounterfactualExplanations

push!(LOAD_PATH, "SklearnBackend")
using SklearnBackend

# generate sample sklearn model
model = gen_sample_model()
# transform it to CounterfactualData
data = CounterfactualData(model)
# transform that back to a sklearn model, just because we can
test = SklearnModel(data, [n for n in model.X.columns], "RandomForestClassifier")
# Select a factual and target
target = 2
factual = 1
x = select_first_factual(data, factual)
generator = GrowingSpheresGenerator()

ce = generate_counterfactual(
  x, target, data, model, generator; 
)
