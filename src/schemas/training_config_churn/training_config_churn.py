from .logistic_regression import LogisticRegressionConfig
from .random_forest import RandomForestConfig

type TrainingConfigChurn = (LogisticRegressionConfig | RandomForestConfig)
