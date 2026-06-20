from typing import Any


class AppError(Exception):
    """Базовое исключение приложения с единым форматом: code, message, details."""

    code: str = "app_error"
    status_code: int = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: Any | None = None,
        status_code: int | None = None,
    ):
        self.message = message
        self.details = details
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)
