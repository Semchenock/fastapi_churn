from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import dataset, health, model
from src.core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    logger.info("Starting churn service")

    app = FastAPI(title="FastAPI Churn Service")

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(dataset.router)
    app.include_router(model.router)

    return app


app = create_app()
