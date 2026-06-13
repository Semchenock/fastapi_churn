import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


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


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
METADATA_PATH = MODEL_DIR / "churn_model_metadata.json"


def save_churn_model(model, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def save_churn_model_metadata(
        metrics: dict[str, float],
        path: Path = METADATA_PATH,
) -> dict:
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "logistic_regression",
        "metrics": metrics,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return metadata


def load_churn_model_metadata(path: Path = METADATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)
