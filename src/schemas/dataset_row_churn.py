from src.schemas.feature_vector_churn import FeatureVectorChurn, FEATURE_VECTOR_CHURN_EXAMPLE
from pydantic import Field, ConfigDict

DATASET_ROW_CHURN_EXAMPLE = {
    **FEATURE_VECTOR_CHURN_EXAMPLE,
    "churn": 1
}

class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., ge=0, le=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": DATASET_ROW_CHURN_EXAMPLE
        }
    )
