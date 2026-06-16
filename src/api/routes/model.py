from fastapi import APIRouter

from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.schemas.prediction_response_churn import PredictionResponseChurn
from src.services.model.model_service import model_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=list[PredictionResponseChurn]
)
def predict(payload: FeatureVectorChurn | list[FeatureVectorChurn]):
    list_payload: list[FeatureVectorChurn] = payload if isinstance(payload, list) else [payload]
    return model_service.predict(list_payload)


@router.post("/model/train")
async def train_model():
    return model_service.train()


@router.get("/model/status")
async def model_status():
    return model_service.status()
