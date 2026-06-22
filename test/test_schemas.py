import pytest
from pydantic import ValidationError

from src.schemas.feature_vector_churn import (
    FeatureVectorChurn,
    FEATURE_VECTOR_CHURN_EXAMPLE,
)
from src.schemas.dataset_row_churn import DatasetRowChurn, DATASET_ROW_CHURN_EXAMPLE
from src.schemas.prediction_response_churn import PredictionResponseChurn
from src.schemas.training_config_churn.logistic_regression import (
    LogisticRegressionConfig,
)
from src.schemas.training_config_churn.random_forest import RandomForestConfig
from src.services.model.enums import ModelTypeEnum


class TestFeatureVectorChurn:
    def test_valid_example(self):
        vector = FeatureVectorChurn(**FEATURE_VECTOR_CHURN_EXAMPLE)
        assert vector.monthly_fee == 9.99
        assert vector.region == "america"

    def test_monthly_fee_must_be_positive(self):
        data = dict(FEATURE_VECTOR_CHURN_EXAMPLE, monthly_fee=0)
        with pytest.raises(ValidationError):
            FeatureVectorChurn(**data)

    def test_negative_usage_hours_rejected(self):
        data = dict(FEATURE_VECTOR_CHURN_EXAMPLE, usage_hours=-1)
        with pytest.raises(ValidationError):
            FeatureVectorChurn(**data)

    def test_autopay_must_be_0_or_1(self):
        data = dict(FEATURE_VECTOR_CHURN_EXAMPLE, autopay_enabled=2)
        with pytest.raises(ValidationError):
            FeatureVectorChurn(**data)

    def test_missing_required_field(self):
        data = dict(FEATURE_VECTOR_CHURN_EXAMPLE)
        del data["region"]
        with pytest.raises(ValidationError):
            FeatureVectorChurn(**data)


class TestDatasetRowChurn:
    def test_valid_example(self):
        row = DatasetRowChurn(**DATASET_ROW_CHURN_EXAMPLE)
        assert row.churn == 1

    @pytest.mark.parametrize("churn", [-1, 2])
    def test_churn_out_of_range(self, churn):
        data = dict(DATASET_ROW_CHURN_EXAMPLE, churn=churn)
        with pytest.raises(ValidationError):
            DatasetRowChurn(**data)


class TestPredictionResponseChurn:
    def test_valid(self):
        response = PredictionResponseChurn(churn=1, probability=0.9)
        assert response.churn == 1
        assert response.probability == 0.9

    def test_negative_probability_rejected(self):
        with pytest.raises(ValidationError):
            PredictionResponseChurn(churn=0, probability=-0.1)


class TestTrainingConfig:
    def test_logistic_regression_defaults(self):
        config = LogisticRegressionConfig()
        assert config.model_type == ModelTypeEnum.LOGISTIC_REGRESSION
        assert config.hyperparameters.solver == "lbfgs"
        assert config.hyperparameters.max_iter == 100

    def test_logistic_regression_invalid_solver(self):
        with pytest.raises(ValidationError):
            LogisticRegressionConfig(hyperparameters={"solver": "unknown"})

    def test_logistic_regression_C_must_be_positive(self):
        with pytest.raises(ValidationError):
            LogisticRegressionConfig(hyperparameters={"C": 0})

    def test_random_forest_defaults(self):
        config = RandomForestConfig()
        assert config.model_type == ModelTypeEnum.RANDOM_FOREST
        assert config.hyperparameters.n_estimators == 100

    def test_random_forest_min_samples_split_validation(self):
        with pytest.raises(ValidationError):
            RandomForestConfig(hyperparameters={"min_samples_split": 1})
