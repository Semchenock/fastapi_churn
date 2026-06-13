from fastapi import FastAPI, Query
from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.services.dataset.dataset_service import dataset_service
from src.model.logistic_regression import train_and_measure_model

app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(payload: FeatureVectorChurn) -> FeatureVectorChurn:
    return payload

@app.get("/preview")
async def preview_dataset(
    limit: int = Query(default=5, ge=1, le=100)
):
    return dataset_service.preview(limit)

@app.get("/info")
async def dataset_info():
    return dataset_service.info()

@app.get("/dataset/split-info")
async def dataset_split_info():
    return dataset_service.split_info()

@app.post("/model/train")
async def train_model():
    _, accuracy, f1  = train_and_measure_model()
    return {"accuracy": accuracy, "f1": f1}
