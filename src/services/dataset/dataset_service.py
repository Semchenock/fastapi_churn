from .loader import DatasetLoader


class DatasetService:

    def __init__(self):
        self.loader = DatasetLoader(
            "data/churn_dataset.csv"
        )

    def preview(self, limit: int = 5):

        df = self.loader.get_dataframe()

        return df.head(limit).to_dict(orient="records")

    def info(self):

        df = self.loader.get_dataframe()

        return {
            "rows": len(df),

            "columns": len(df.columns),

            "features": list(df.columns),

            "churn_distribution": (
                df["churn"]
                .value_counts()
                .to_dict()
            )
        }


dataset_service = DatasetService()
