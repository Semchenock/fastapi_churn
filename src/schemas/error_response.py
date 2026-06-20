from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Машиночитаемый код ошибки")
    message: str = Field(..., description="Человекочитаемое описание ошибки")
    details: Any | None = Field(
        default=None, description="Дополнительный контекст ошибки"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "model_error",
                "message": "Model is not trained",
                "details": None,
            }
        }
    }
