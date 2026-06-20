from fastapi import APIRouter, Query

from src.services.dataset.dataset_service import dataset_service

router = APIRouter()


@router.get("/dataset/preview")
async def preview_dataset(limit: int = Query(default=5, ge=1, le=100)):
    return dataset_service.preview(limit)


@router.get("/dataset/info")
async def dataset_info():
    return dataset_service.info()


@router.get("/dataset/split-info")
async def dataset_split_info():
    return dataset_service.split_info()
