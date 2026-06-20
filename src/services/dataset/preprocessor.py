import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.services.dataset.exceptions import DatasetError


class DatasetPreprocessor:
    NUMERIC_COLUMNS = [
        "monthly_fee",
        "usage_hours",
        "support_requests",
        "account_age_months",
        "failed_payments",
        "autopay_enabled"
    ]

    CATEGORICAL_COLUMNS = [
        "region",
        "device_type",
        "payment_method"
    ]

    def drop_rows_without_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["churn"])

        if df.empty:
            raise DatasetError("Dataset is empty")

        return df

    def split_features_and_target(self, df):
        features = df.drop(columns=["churn"])
        target = df["churn"]
        return features, target

    def get_feature_types(self):
        return {
            "numeric": self.NUMERIC_COLUMNS,
            "categorical": self.CATEGORICAL_COLUMNS,
        }

    def split_train_test(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target
        )

        return X_train, X_test, y_train, y_test

    def get_split_info(self, y_train, y_test):
        return {
            "train": {
                **y_train.value_counts(normalize=True).to_dict(),
                "total": len(y_train),
            },
            "test": {
                **y_test.value_counts(normalize=True).to_dict(),
                "total": len(y_test),
            }
        }

    def prepare_train_test_split(self, df):
        if df.empty:
            raise DatasetError("Dataset is empty")

        prepared_df = self.drop_rows_without_target(df)
        features, target = self.split_features_and_target(prepared_df)
        X_train, X_test, y_train, y_test = self.split_train_test(features, target)

        return X_train, X_test, y_train, y_test

    def build_preprocessor(self):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.NUMERIC_COLUMNS),
            ("cat", categorical_transformer, self.CATEGORICAL_COLUMNS)
        ])

        return preprocessor
