from fastapi import APIRouter

from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.services.model.model_service import model_service

router = APIRouter()


@router.post("/predict")
def predict(payload: FeatureVectorChurn):
    return model_service.predict(payload)


@router.post("/model/train")
async def train_model():
    return model_service.train()
