import pandas as pd

from src.core.logging_config import get_logger
from src.model.logistic_regression import (
    load_churn_model,
    load_churn_model_history,
    load_churn_model_metadata,
)
from src.schemas.feature_vector_churn import FeatureVectorChurn

from src.schemas.prediction_response_churn import PredictionResponseChurn
from src.schemas.training_config_churn import TrainingConfigChurn
from src.services.model.exceptions import ModelError
from src.services.model.training_service import training_service

logger = get_logger(__name__)


class ModelService:
    def __init__(self):
        self.metrics: dict[str, float] | None = None
        self.trained_at: str | None = None
        self.model_type: str | None = None
        self.config: dict | None = None
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = load_churn_model()
            self._load_metadata()
            logger.info("Churn model loaded from disk")
            return model
        except FileNotFoundError:
            logger.warning("Churn model artifact not found, service starts untrained")
            return None

    def _load_metadata(self) -> None:
        try:
            self._apply_metadata(load_churn_model_metadata())
        except FileNotFoundError:
            self.metrics = None
            self.trained_at = None

    def _apply_metadata(self, metadata: dict) -> None:
        self.metrics = metadata.get("metrics")
        self.trained_at = metadata.get("trained_at")
        self.model_type = metadata.get("model_type")
        self.config = metadata.get("config")

    def _validate_features(self, vectors: list[FeatureVectorChurn]):
        features_schema = self.schema()
        expected_names = {field["name"] for field in features_schema}
        type_checks = {
            "integer": int,
            "float": (int, float),
            "string": str,
            "boolean": bool,
        }

        for index, vector in enumerate(vectors):
            values = vector.model_dump()

            # неверное количество признаков
            actual_names = set(values.keys())
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                raise ModelError(
                    f"Vector #{index}: invalid feature set",
                    code="invalid_feature_count",
                    details={"index": index, "missing": missing, "extra": extra},
                )

            # неверные типы значений
            for field in features_schema:
                name = field["name"]
                expected_type = type_checks.get(field["type"])
                value = values[name]

                if expected_type is None:
                    continue

                # bool является подклассом int — исключаем его для числовых полей
                if field["type"] in ("integer", "float") and isinstance(value, bool):
                    raise ModelError(
                        f"Vector #{index}: field '{name}' has invalid type",
                        code="invalid_feature_type",
                        details={
                            "index": index,
                            "field": name,
                            "expected": field["type"],
                            "got": "bool",
                        },
                    )

                if not isinstance(value, expected_type):
                    raise ModelError(
                        f"Vector #{index}: field '{name}' has invalid type",
                        code="invalid_feature_type",
                        details={
                            "index": index,
                            "field": name,
                            "expected": field["type"],
                            "got": type(value).__name__,
                        },
                    )

    def train(self, config: TrainingConfigChurn) -> dict[str, float]:
        self.model, metadata = training_service.train(config)
        self._apply_metadata(metadata)
        return self.metrics

    def schema(self) -> list[dict]:
        type_names = {int: "integer", float: "float", str: "string", bool: "boolean"}

        features = [
            {
                "name": name,
                "type": type_names.get(field.annotation, str(field.annotation)),
                "required": field.is_required(),
            }
            for name, field in FeatureVectorChurn.model_fields.items()
        ]

        return features

    def status(self) -> dict:
        return {
            "is_trained": self.model is not None,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "model_type": self.model_type,
            "config": self.config.get("hyperparameters") if self.config is not None else None,
        }

    def predict(self, payload: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
        logger.info("Predict called for %d vector(s)", len(payload))

        if self.model is None:
            logger.warning("Predict rejected: model is not trained")
            raise ModelError("Model is not trained")

        self._validate_features(payload)

        try:
            features = pd.DataFrame([p.model_dump() for p in payload])
        except Exception as exc:
            raise ModelError(
                "Failed to prepare features for prediction",
                code="data_preparation_error",
                details={"reason": str(exc)},
            ) from exc

        try:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
        except Exception as exc:
            raise ModelError(
                "Model failed to produce a prediction",
                code="prediction_error",
                details={"reason": str(exc)},
            ) from exc

        logger.info("Predict succeeded for %d vector(s)", len(payload))
        return [
            PredictionResponseChurn(churn=int(pred), probability=float(prob[pred]))
            for pred, prob in zip(predictions, probabilities)
        ]

    def get_metrics(self, model_type: str | None = None, limit: int | None = None) -> dict:
        history = load_churn_model_history()

        if model_type is not None:
            history = [record for record in history if record.get("model_type") == model_type]

        latest = history[-1] if history else None

        recent = list(reversed(history))
        if limit is not None:
            recent = recent[:limit]

        return {
            "latest": latest,
            "history": recent,
        }


model_service = ModelService()
