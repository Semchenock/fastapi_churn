from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from src.services.model.enums import ModelTypeEnum

RANDOM_FOREST_HYPERPARAMETERS_EXAMPLE = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
}


class RandomForestHyperParameters(BaseModel):
    n_estimators: int = Field(default=100, gt=0)
    criterion: Literal["gini", "entropy", "log_loss"] = Field(default="gini")
    max_depth: Optional[int] = Field(default=None, gt=0)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features: Optional[Union[Literal["sqrt", "log2"], float]] = Field(
        default="sqrt"
    )
    bootstrap: bool = Field(default=True)
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = Field(
        default=None
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": RANDOM_FOREST_HYPERPARAMETERS_EXAMPLE
        }
    )


class RandomForestConfig(BaseModel):
    model_type: Literal[ModelTypeEnum.RANDOM_FOREST] = (
        ModelTypeEnum.RANDOM_FOREST
    )
    hyperparameters: RandomForestHyperParameters = Field(
        default_factory=RandomForestHyperParameters
    )
