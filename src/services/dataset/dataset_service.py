from .loader import DatasetLoader
from .preprocessor import DatasetPreprocessor
from .exceptions import DatasetError


class DatasetService:

    def __init__(self):
        self.loader = DatasetLoader(
            "data/churn_dataset.csv"
        )
        self.preprocessor = DatasetPreprocessor()
        self._train_test = None

    def preview(self, limit: int = 5):

        df = self.loader.get_dataframe()
        self.ensure_not_empty(df)

        return df.head(limit).to_dict(orient="records")

    def info(self):

        df = self.loader.get_dataframe()
        self.ensure_not_empty(df)

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
        X_train, X_test, y_train, y_test = self.get_train_test()
        self.ensure_not_empty(X_train)

        return self.preprocessor.get_split_info(y_train, y_test)

    def get_processor(self):
        return self.preprocessor.build_preprocessor()

    def get_train_test(self):
        if self._train_test is not None:
            return self._train_test

        df = self.loader.get_dataframe()
        self.ensure_not_empty(df)
        self._train_test = self.preprocessor.preprocess(df)
        return self._train_test

    def get_train(self):
        X_train, _, y_train, _ = self.get_train_test()
        return X_train, y_train

    def get_test(self):
        _, X_test, _, y_test = self.get_train_test()
        return X_test, y_test

    def ensure_not_empty(self, df):
        if df.empty:
            raise DatasetError("Dataset is empty")


dataset_service = DatasetService()
