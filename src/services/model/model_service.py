import pandas as pd

from src.model.logistic_regression import (
    load_churn_model,
    load_churn_model_metadata,
)
from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.schemas.prediction_response_churn import PredictionResponseChurn
from src.schemas.training_config_churn import TrainingConfigChurn
from src.services.model.exceptions import ModelError
from src.services.model.training_service import training_service


class ModelService:
    def __init__(self):
        self.metrics: dict[str, float] | None = None
        self.trained_at: str | None = None
        self.model = self._load_model()
        self.model_type: str | None = None
        self.config: dict | None = None

    def _load_model(self):
        try:
            model = load_churn_model()
            self._load_metadata()
            return model
        except FileNotFoundError:
            # TODO remove auto training
            model, metadata = training_service.train()
            self._apply_metadata(metadata)
            return model

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

    def train(self, config: TrainingConfigChurn) -> dict[str, float]:
        self.model, metadata = training_service.train(config)
        self._apply_metadata(metadata)
        return self.metrics

    def status(self) -> dict:
        return {
            "is_trained": self.model is not None,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "model_type": self.model_type,
            "config": self.config.get("hyperparameters") if self.config is not None else None,
        }

    def predict(self, payload: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
        if self.model is None:
            raise ModelError("Model is not trained")

        features = pd.DataFrame([p.model_dump() for p in payload])
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        return [
            PredictionResponseChurn(churn=int(pred), probability=float(prob[pred]))
            for pred, prob in zip(predictions, probabilities)
        ]


model_service = ModelService()
