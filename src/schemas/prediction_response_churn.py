from pydantic import BaseModel, Field, ConfigDict

PREDICTION_RESPONSE_CHURN_EXAMPLE = {
    "churn": 1,
    "probabilities": {"0": 0.01, "1": 0.99},
}

class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., ge=0)
    probabilities: dict[str, float] = Field(...)

    model_config = ConfigDict(
        json_schema_extra={
            "example": PREDICTION_RESPONSE_CHURN_EXAMPLE
        }
    )

