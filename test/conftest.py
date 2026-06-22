import pandas as pd
import pytest

from src.schemas.dataset_row_churn import DATASET_ROW_CHURN_EXAMPLE


def _make_row(**overrides) -> dict:
    row = dict(DATASET_ROW_CHURN_EXAMPLE)
    row.update(overrides)
    return row


@pytest.fixture
def sample_records() -> list[dict]:
    """Небольшой сбалансированный по churn набор строк датасета."""
    return [
        _make_row(churn=1, region="america"),
        _make_row(churn=0, region="europe", monthly_fee=19.99),
        _make_row(churn=1, region="asia", usage_hours=5.0),
        _make_row(churn=0, region="america", account_age_months=40),
        _make_row(churn=1, device_type="mobile"),
        _make_row(churn=0, payment_method="paypal"),
        _make_row(churn=1, failed_payments=3),
        _make_row(churn=0, autopay_enabled=0),
        _make_row(churn=1, support_requests=5),
        _make_row(churn=0, monthly_fee=49.99),
    ]


@pytest.fixture
def sample_dataframe(sample_records) -> pd.DataFrame:
    return pd.DataFrame(sample_records)


@pytest.fixture
def csv_path(tmp_path, sample_dataframe):
    path = tmp_path / "churn.csv"
    sample_dataframe.to_csv(path, index=False)
    return str(path)
