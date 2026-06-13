import pandas as pd

from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.services.model.exceptions import ModelError
from src.services.model.training_service import training_service


class ModelService:
    def __init__(self):
        self.model = None
        self.metrics: dict[str, float] | None = None

    def train(self) -> dict[str, float]:
        self.model, self.metrics = training_service.train()
        return self.metrics

    def predict(self, payload: FeatureVectorChurn) -> dict[str, int]:
        if self.model is None:
            raise ModelError("Model is not trained")

        features = pd.DataFrame([payload.model_dump()])
        prediction = self.model.predict(features)[0]

        return {"churn": int(prediction)}


model_service = ModelService()
