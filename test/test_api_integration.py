"""Интеграционные тесты сценария обучение → статус → предсказание.

Чтобы тесты были повторяемыми и не зависели от реального
data/churn_dataset.csv и каталога models/, синглтоны переключаются на
синтетический датасет во временной директории, а все операции
сохранения/загрузки модели перенаправляются в tmp_path.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.services.model.model_service as model_service_module
import src.services.model.training_service as training_service_module
from src.api.app import app
from src.model.logistic_regression import model as model_module
from src.schemas.feature_vector_churn import FEATURE_VECTOR_CHURN_EXAMPLE
from src.services.dataset.dataset_service import dataset_service
from src.services.dataset.loader import DatasetLoader
from src.services.model.model_service import model_service

# Признаки с явным сигналом: churn=1 — дорогой тариф, мало активности,
# просроченные платежи, без автоплатежа; churn=0 — наоборот.
_CHURN_ROW = {
    "monthly_fee": 49.99,
    "usage_hours": 3.0,
    "support_requests": 5,
    "account_age_months": 2,
    "failed_payments": 3,
    "region": "america",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0,
    "churn": 1,
}

_STAY_ROW = {
    "monthly_fee": 9.99,
    "usage_hours": 40.0,
    "support_requests": 0,
    "account_age_months": 36,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1,
    "churn": 0,
}


def _synthetic_records(n_per_class: int = 15) -> list[dict]:
    """Детерминированный сбалансированный набор с лёгким разбросом значений."""
    records = []
    for i in range(n_per_class):
        churn = dict(_CHURN_ROW)
        churn["usage_hours"] = 3.0 + i * 0.1
        churn["monthly_fee"] = 49.99 - i * 0.1

        stay = dict(_STAY_ROW)
        stay["usage_hours"] = 40.0 - i * 0.1
        stay["account_age_months"] = 36 - i

        records.append(churn)
        records.append(stay)
    return records


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    df = pd.DataFrame(_synthetic_records())
    csv_path = tmp_path / "churn_dataset.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(dataset_service, "loader", DatasetLoader(str(csv_path)))
    monkeypatch.setattr(dataset_service, "_train_test", None)

    model_path = tmp_path / "model.pkl"
    meta_path = tmp_path / "meta.json"
    history_path = tmp_path / "history.json"

    monkeypatch.setattr(
        training_service_module,
        "save_churn_model",
        lambda model: model_module.save_churn_model(model, path=model_path),
    )
    monkeypatch.setattr(
        training_service_module,
        "save_churn_model_metadata",
        lambda metrics, config: model_module.save_churn_model_metadata(
            metrics, config, path=meta_path, history_path=history_path
        ),
    )
    monkeypatch.setattr(
        model_service_module,
        "load_churn_model",
        lambda path=model_path: model_module.load_churn_model(path=path),
    )
    monkeypatch.setattr(
        model_service_module,
        "load_churn_model_metadata",
        lambda path=meta_path: model_module.load_churn_model_metadata(path=path),
    )
    monkeypatch.setattr(
        model_service_module,
        "load_churn_model_history",
        lambda path=history_path: model_module.load_churn_model_history(path=path),
    )

    monkeypatch.setattr(model_service, "model", None)
    monkeypatch.setattr(model_service, "metrics", None)
    monkeypatch.setattr(model_service, "trained_at", None)
    monkeypatch.setattr(model_service, "model_type", None)
    monkeypatch.setattr(model_service, "config", None)

    with TestClient(app) as client:
        yield client


class TestDatasetEndpoints:
    def test_preview_reads_dataset(self, isolated_client):
        response = isolated_client.get("/dataset/preview", params={"limit": 4})
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 4
        assert "churn" in body[0]

    def test_info_reports_rows_and_distribution(self, isolated_client):
        response = isolated_client.get("/dataset/info")
        assert response.status_code == 200
        body = response.json()
        assert body["rows"] == 30
        assert set(body["churn_distribution"].keys()) == {"0", "1"}


class TestTrainStatusPredictFlow:
    def test_status_before_training(self, isolated_client):
        response = isolated_client.get("/model/status")
        assert response.status_code == 200
        assert response.json()["is_trained"] is False

    def test_predict_before_training_fails(self, isolated_client):
        response = isolated_client.post("/predict", json=FEATURE_VECTOR_CHURN_EXAMPLE)
        assert response.status_code == 400
        assert response.json()["code"] == "model_error"

    def test_full_flow(self, isolated_client):
        train_response = isolated_client.post(
            "/model/train", json={"model_type": "logistic_regression"}
        )
        assert train_response.status_code == 200
        metrics = train_response.json()
        assert "accuracy" in metrics and "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

        status_response = isolated_client.get("/model/status")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["is_trained"] is True
        assert status["model_type"] == "logistic_regression"
        assert status["trained_at"] is not None

        predict_response = isolated_client.post(
            "/predict", json=FEATURE_VECTOR_CHURN_EXAMPLE
        )
        assert predict_response.status_code == 200
        predictions = predict_response.json()
        assert len(predictions) == 1
        assert predictions[0]["churn"] in (0, 1)
        probabilities = predictions[0]["probabilities"]
        assert set(probabilities) == {"0", "1"}
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())

    def test_predict_batch_after_training(self, isolated_client):
        isolated_client.post("/model/train", json={"model_type": "logistic_regression"})

        churn_like = dict(_CHURN_ROW)
        churn_like.pop("churn")
        stay_like = dict(_STAY_ROW)
        stay_like.pop("churn")

        response = isolated_client.post("/predict", json=[churn_like, stay_like])
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_random_forest_training_flow(self, isolated_client):
        train_response = isolated_client.post(
            "/model/train", json={"model_type": "random_forest"}
        )
        assert train_response.status_code == 200

        status = isolated_client.get("/model/status").json()
        assert status["model_type"] == "random_forest"


class TestValidationAndMetrics:
    def test_predict_invalid_body_returns_422(self, isolated_client):
        isolated_client.post("/model/train", json={"model_type": "logistic_regression"})
        bad = dict(FEATURE_VECTOR_CHURN_EXAMPLE, monthly_fee=-5)
        response = isolated_client.post("/predict", json=bad)
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_train_invalid_model_type_returns_422(self, isolated_client):
        response = isolated_client.post("/model/train", json={"model_type": "svm"})
        assert response.status_code == 422

    def test_metrics_history_grows_with_training(self, isolated_client):
        isolated_client.post("/model/train", json={"model_type": "logistic_regression"})
        isolated_client.post("/model/train", json={"model_type": "random_forest"})

        response = isolated_client.get("/model/metrics")
        assert response.status_code == 200
        body = response.json()
        assert len(body["history"]) == 2
        assert body["latest"]["model_type"] == "random_forest"
