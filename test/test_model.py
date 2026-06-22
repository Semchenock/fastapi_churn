import json

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.model.logistic_regression import model as model_module
from src.model.logistic_regression.model import (
    build_pipeline,
    evaluate_model,
    get_classifier,
    load_churn_model,
    load_churn_model_history,
    load_churn_model_metadata,
    save_churn_model,
    save_churn_model_metadata,
    train_churn_model,
)
from src.schemas.training_config_churn.logistic_regression import (
    LogisticRegressionConfig,
)
from src.schemas.training_config_churn.random_forest import RandomForestConfig
from src.services.dataset.preprocessor import DatasetPreprocessor


class TestGetClassifier:
    def test_logistic_regression(self):
        clf = get_classifier(LogisticRegressionConfig())
        assert isinstance(clf, LogisticRegression)

    def test_random_forest(self):
        clf = get_classifier(RandomForestConfig())
        assert isinstance(clf, RandomForestClassifier)

    def test_hyperparameters_applied(self):
        config = LogisticRegressionConfig(hyperparameters={"max_iter": 250})
        clf = get_classifier(config)
        assert clf.max_iter == 250


class TestPipeline:
    def _split(self, df):
        pre = DatasetPreprocessor()
        return pre.prepare_train_test_split(df), pre.build_preprocessor()

    def test_build_pipeline(self, sample_dataframe):
        (_, _, _, _), preprocessor = self._split(sample_dataframe)
        pipeline = build_pipeline(preprocessor, LogisticRegressionConfig())
        assert isinstance(pipeline, Pipeline)

    def test_train_and_evaluate(self, sample_dataframe):
        (X_train, X_test, y_train, y_test), preprocessor = self._split(
            sample_dataframe
        )
        pipeline = build_pipeline(preprocessor, LogisticRegressionConfig())
        trained = train_churn_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained, X_test, y_test)
        assert set(metrics) == {"accuracy", "f1"}
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestModelPersistence:
    def _trained_pipeline(self, sample_dataframe):
        pre = DatasetPreprocessor()
        X_train, _, y_train, _ = pre.prepare_train_test_split(sample_dataframe)
        pipeline = build_pipeline(pre.build_preprocessor(), LogisticRegressionConfig())
        return train_churn_model(pipeline, X_train, y_train)

    def test_save_and_load_model(self, tmp_path, sample_dataframe):
        path = tmp_path / "model.pkl"
        model = self._trained_pipeline(sample_dataframe)
        save_churn_model(model, path=path)
        assert path.exists()
        assert isinstance(load_churn_model(path=path), Pipeline)

    def test_load_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_churn_model(path=tmp_path / "missing.pkl")

    def test_save_and_load_metadata(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        history_path = tmp_path / "history.json"
        metrics = {"accuracy": 0.8, "f1": 0.7}
        config = LogisticRegressionConfig()

        metadata = save_churn_model_metadata(
            metrics, config, path=meta_path, history_path=history_path
        )
        assert metadata["metrics"] == metrics
        assert load_churn_model_metadata(path=meta_path)["metrics"] == metrics

    def test_metadata_history_appends(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        history_path = tmp_path / "history.json"
        config = LogisticRegressionConfig()

        save_churn_model_metadata(
            {"accuracy": 0.5, "f1": 0.5}, config,
            path=meta_path, history_path=history_path,
        )
        save_churn_model_metadata(
            {"accuracy": 0.9, "f1": 0.9}, config,
            path=meta_path, history_path=history_path,
        )
        history = load_churn_model_history(path=history_path)
        assert len(history) == 2

    def test_load_missing_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_churn_model_metadata(path=tmp_path / "missing.json")

    def test_load_missing_history_returns_empty(self, tmp_path):
        assert load_churn_model_history(path=tmp_path / "missing.json") == []
