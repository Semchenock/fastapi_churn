from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from src.services.model.enums import ModelTypeEnum

LOGISTIC_REGRESSION_HYPERPARAMETERS_EXAMPLE = {
    "penalty": "l2",
    "tol": 0.0001,
    "C": 1.0,
    "fit_intercept": True,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 100,
}


class LogisticRegressionHyperParameters(BaseModel):
    penalty: Optional[Literal["l1", "l2", "elasticnet"]] = Field(default="l2")
    tol: float = Field(default=1e-4, gt=0)
    C: float = Field(default=1.0, gt=0)
    fit_intercept: bool = Field(default=True)
    class_weight: Optional[Literal["balanced"]] = Field(default=None)
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = Field(default="lbfgs")
    max_iter: int = Field(default=100, gt=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": LOGISTIC_REGRESSION_HYPERPARAMETERS_EXAMPLE
        }
    )


class LogisticRegressionConfig(BaseModel):
    model_type: Literal[ModelTypeEnum.LOGISTIC_REGRESSION] = (
        ModelTypeEnum.LOGISTIC_REGRESSION
    )
    hyperparameters: LogisticRegressionHyperParameters = Field(
        default_factory=LogisticRegressionHyperParameters
    )
