from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import dataset, health, model


def create_app() -> FastAPI:
    app = FastAPI()

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(dataset.router)
    app.include_router(model.router)

    return app


app = create_app()
