class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., ge=0, le=1)
