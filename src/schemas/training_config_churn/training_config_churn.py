from typing import Annotated, Union

from pydantic import Field

from .logistic_regression import LogisticRegressionConfig
from .random_forest import RandomForestConfig

type TrainingConfigChurn = Annotated[
    Union[LogisticRegressionConfig, RandomForestConfig],
    Field(discriminator="model_type"),
]
