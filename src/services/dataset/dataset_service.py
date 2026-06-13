from .loader import DatasetLoader
from .preprocessor import DatasetPreprocessor


class DatasetService:

    def __init__(self):
        self.loader = DatasetLoader(
            "data/churn_dataset.csv"
        )
        self.preprocessor = DatasetPreprocessor()

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

    def split_info(self):
        df = self.loader.get_dataframe()

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(df)

        return self.preprocessor.get_split_info(y_train, y_test)


dataset_service = DatasetService()
