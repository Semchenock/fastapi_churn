from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def build_pipeline(preprocessor) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])


def train_churn_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict[str, float]:
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return {"accuracy": float(accuracy), "f1": float(f1)}
