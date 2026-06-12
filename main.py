from fastapi import FastAPI
from src.schemas.FeatureVectorChurn import FeatureVectorChurn

app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(payload: FeatureVectorChurn) -> FeatureVectorChurn:
    return payload
