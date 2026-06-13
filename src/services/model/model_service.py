import pandas as pd

from src.model.logistic_regression import (
    load_churn_model,
    load_churn_model_metadata,
)
from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.services.model.exceptions import ModelError
from src.services.model.training_service import training_service


class ModelService:
    def __init__(self):
        self.metrics: dict[str, float] | None = None
        self.trained_at: str | None = None
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = load_churn_model()
            self._load_metadata()
            return model
        except FileNotFoundError:
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

    def train(self) -> dict[str, float]:
        self.model, metadata = training_service.train()
        self._apply_metadata(metadata)
        return self.metrics

    def status(self) -> dict:
        return {
            "is_trained": self.model is not None,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }

    def predict(self, payload: FeatureVectorChurn) -> dict[str, int]:
        if self.model is None:
            raise ModelError("Model is not trained")

        features = pd.DataFrame([payload.model_dump()])
        prediction = self.model.predict(features)[0]

        return {"churn": int(prediction)}


model_service = ModelService()
