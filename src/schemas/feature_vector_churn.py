from pydantic import BaseModel, Field, ConfigDict

FEATURE_VECTOR_CHURN_EXAMPLE = {
  "monthly_fee": 9.99,
  "usage_hours": 27.92,
  "support_requests": 1,
  "account_age_months": 14,
  "failed_payments": 1,
  "region": "america",
  "device_type": "desktop",
  "payment_method": "card",
  "autopay_enabled": 1
}


class FeatureVectorChurn(BaseModel):
    monthly_fee: float = Field(..., gt=0)
    usage_hours: float = Field(..., ge=0)

    support_requests: int = Field(..., ge=0)
    account_age_months: int = Field(..., ge=0)
    failed_payments: int = Field(..., ge=0)

    region: str
    device_type: str
    payment_method: str

    autopay_enabled: int = Field(..., ge=0, le=1)

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "example": FEATURE_VECTOR_CHURN_EXAMPLE
        }
    )
