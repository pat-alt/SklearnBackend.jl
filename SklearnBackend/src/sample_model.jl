function gen_sample_model()
py"""
from sklearn.compose import ColumnTransformer
import sklearn.datasets as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def gen_sample_sklearn_model():
    dataset = sd.load_iris(as_frame=True).frame
    target = dataset["target"]
    data_without_target = dataset.drop("target", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        data_without_target, target, test_size=0.2, random_state=0, stratify=target
    )
    numerical = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    categorical = x_train.columns.difference(numerical)
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
        steps=[("preprocessor", transformations), ("classifier", RandomForestClassifier())]
    )
    model = clf.fit(x_train, y_train)
    return model, x_train, y_train, categorical, numerical
"""
    model, X, y, categorical, numerical = py"gen_sample_sklearn_model()"
    SklearnModel(model, X, y, categorical, numerical, :classification_multi)
end
