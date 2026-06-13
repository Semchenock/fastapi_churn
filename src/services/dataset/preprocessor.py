import pandas as pd
from sklearn.model_selection import train_test_split

from src.schemas.dataset_row_churn import DatasetRowChurn

MEAN_COLUMNS = [
    "monthly_fee",
    "usage_hours"
]

MEAN_COLUMNS_INT = [
    "support_requests",
    "account_age_months",
    "failed_payments",
]

MODE_COLUMNS = [
    "region",
    "device_type",
    "payment_method"
]

BINARY_COLUMNS = [
    "autopay_enabled"
]

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


class DatasetPreprocessor:
    def fill_missing_values(self, df):
        df = df.dropna(subset=["churn"])

        for column in MEAN_COLUMNS:
            df[column] = df[column].fillna(df[column].mean())

        for column in MEAN_COLUMNS_INT:
            df[column] = df[column].fillna(int(df[column].mean()))

        for column in MODE_COLUMNS:
            df[column] = df[column].fillna(df[column].mode()[0])

        for column in BINARY_COLUMNS:
            df[column] = df[column].fillna(0)

        return df

    def split_features_and_target(self, df):
        features = df.drop(columns=["churn"])
        target = df["churn"]
        return features, target

    def get_feature_types(self):
        return {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
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
            "train": y_train.value_counts(normalize=True).to_dict(),
            "test": y_test.value_counts(normalize=True).to_dict()
        }

    def preprocess(self, df):
        prepared_df = self.fill_missing_values(df)
        features, target = self.split_features_and_target(prepared_df)
        X_train, X_test, y_train, y_test = self.split_train_test(features, target)

        return X_train, X_test, y_train, y_test