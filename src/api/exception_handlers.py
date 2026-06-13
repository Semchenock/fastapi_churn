from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.services.dataset.exceptions import DatasetError
from src.services.model.exceptions import ModelError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DatasetError)
    async def dataset_error_handler(request: Request, exc: DatasetError):
        return JSONResponse(
            status_code=400,
            content={"detail": exc.message},
        )

    @app.exception_handler(ModelError)
    async def model_error_handler(request: Request, exc: ModelError):
        return JSONResponse(
            status_code=400,
            content={"detail": exc.message},
        )
