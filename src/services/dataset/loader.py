import pandas as pd
from pandas.errors import EmptyDataError

from src.core.logging_config import get_logger
from src.schemas.dataset_row_churn import DatasetRowChurn
from .exceptions import DatasetError

logger = get_logger(__name__)


class DatasetLoader:

    def __init__(self, path: str):
        self.path = path
        self.df: pd.DataFrame | None = None

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            self.df = self._load_dataframe()

        return self.df

    def _load_dataframe(self) -> pd.DataFrame:
        logger.info("Loading churn dataset from %s", self.path)
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError as exc:
            logger.error("Churn dataset not found at %s", self.path)
            raise DatasetError("Dataset is not loaded") from exc
        except EmptyDataError as exc:
            logger.error("Churn dataset at %s is empty", self.path)
            raise DatasetError("Dataset is empty") from exc

        if df.empty:
            logger.error("Churn dataset at %s is empty", self.path)
            raise DatasetError("Dataset is empty")

        logger.info(
            "Churn dataset loaded: %d rows, %d columns", len(df), len(df.columns)
        )
        return df

    def get_rows(self) -> list[DatasetRowChurn]:
        df = self.get_dataframe()

        rows = []

        for record in df.to_dict(orient="records"):
            rows.append(DatasetRowChurn(**record))

        return rows
