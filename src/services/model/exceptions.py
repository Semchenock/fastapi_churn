from src.services.exceptions import AppError


class ModelError(AppError):
    code = "model_error"
