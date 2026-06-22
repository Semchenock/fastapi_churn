from pathlib import Path

from fastapi import APIRouter

from src.services.dataset.dataset_service import dataset_service
from src.services.model.model_service import model_service

router = APIRouter()


@router.get("/")
def read_root() -> dict[str, str]:
    return {"message": "ml churn service is running"}


@router.get("/health")
def health() -> dict:
    model_available = model_service.model is not None

    loader = dataset_service.loader
    dataset_loaded = loader.df is not None or Path(loader.path).exists()

    status = "ok" if model_available and dataset_loaded else "degraded"

    return {
        "status": status,
        "model_available": model_available,
        "dataset_loaded": dataset_loaded,
    }
