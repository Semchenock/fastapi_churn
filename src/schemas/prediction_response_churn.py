from pydantic import BaseModel, Field, ConfigDict

PREDICTION_RESPONSE_CHURN_EXAMPLE = {
    "churn": 1,
    "probability": 0.99
}

class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., ge=0)
    probability: float = Field(..., ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": PREDICTION_RESPONSE_CHURN_EXAMPLE
        }
    )

