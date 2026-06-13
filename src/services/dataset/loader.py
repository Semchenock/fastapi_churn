import pandas as pd
from pandas.errors import EmptyDataError

from src.schemas.dataset_row_churn import DatasetRowChurn
from .exceptions import DatasetError


class DatasetLoader:

    def __init__(self, path: str):
        self.path = path
        self.df: pd.DataFrame | None = None

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            self.df = self._load_dataframe()

        return self.df

    def _load_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError as exc:
            raise DatasetError("Dataset is not loaded") from exc
        except EmptyDataError as exc:
            raise DatasetError("Dataset is empty") from exc

        if df.empty:
            raise DatasetError("Dataset is empty")

        return df

    def get_rows(self) -> list[DatasetRowChurn]:
        df = self.get_dataframe()

        rows = []

        for record in df.to_dict(orient="records"):
            rows.append(DatasetRowChurn(**record))

        return rows
