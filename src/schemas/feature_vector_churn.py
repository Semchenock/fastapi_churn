from pydantic import BaseModel, Field


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
