from src.services.exceptions import AppError


class DatasetError(AppError):
    code = "dataset_error"
