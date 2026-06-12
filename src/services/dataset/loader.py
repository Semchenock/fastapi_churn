import pandas as pd

from src.schemas.dataset_row_churn import DatasetRowChurn


class DatasetLoader:

    def __init__(self, path: str):
        self.path = path

        self.df = pd.read_csv(path)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_rows(self) -> list[DatasetRowChurn]:

        rows = []

        for record in self.df.to_dict(orient="records"):
            rows.append(DatasetRowChurn(**record))

        return rows
