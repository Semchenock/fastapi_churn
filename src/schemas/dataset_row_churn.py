from src.schemas.feature_vector_churn import FeatureVectorChurn
from pydantic import Field


class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., ge=0, le=1)
