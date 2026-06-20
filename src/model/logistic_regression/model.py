import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.schemas.training_config_churn import TrainingConfigChurn
from src.services.model.enums import ModelTypeEnum
from src.services.model.exceptions import ModelError

type ClassifierType = (LogisticRegression | RandomForestClassifier)

def get_classifier(config: TrainingConfigChurn) -> ClassifierType:
    if config.model_type == ModelTypeEnum.LOGISTIC_REGRESSION:
        return LogisticRegression(**config.hyperparameters.model_dump())

    elif config.model_type == ModelTypeEnum.RANDOM_FOREST:
        return RandomForestClassifier(**config.hyperparameters.model_dump())

    raise ModelError(f"Invalid model type: {config.model_type}")


def build_pipeline(preprocessor, config) -> Pipeline:
    classifier = get_classifier(config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
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
METADATA_HISTORY_PATH = MODEL_DIR / "churn_model_history.json"


def save_churn_model(model, path: Path = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_churn_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def save_churn_model_metadata(
        metrics: dict[str, float],
        config: TrainingConfigChurn,
        path: Path = METADATA_PATH,
        history_path: Path = METADATA_HISTORY_PATH,
) -> dict:
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": config.model_type,
        "metrics": metrics,
        "config": config.model_dump(mode="json"),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    history = []

    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)

    history.append(metadata)

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    return metadata


def load_churn_model_metadata(path: Path = METADATA_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)
