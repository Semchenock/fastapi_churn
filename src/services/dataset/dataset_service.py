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

    def get_processor(self):
        return self.preprocessor.build_preprocessor()

    def get_train_test(self):
        df = self.loader.get_dataframe()
        features, target = self.preprocessor.split_features_and_target(df)
        X_train, X_test, y_train, y_test = self.preprocessor.split_train_test(features, target)
        return X_train, X_test, y_train, y_test

    def get_train(self):
        X_train, _, y_train = self.get_train_test()
        return X_train, y_train

    def get_test(self):
        _, X_test, _, y_test = self.get_train_test()
        return X_test, y_test


dataset_service = DatasetService()