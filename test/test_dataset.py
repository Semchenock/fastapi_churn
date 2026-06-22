import pandas as pd
import pytest

from src.schemas.dataset_row_churn import DatasetRowChurn
from src.services.dataset.loader import DatasetLoader
from src.services.dataset.preprocessor import DatasetPreprocessor
from src.services.dataset.exceptions import DatasetError


class TestDatasetLoader:
    def test_get_dataframe(self, csv_path, sample_records):
        loader = DatasetLoader(csv_path)
        df = loader.get_dataframe()
        assert len(df) == len(sample_records)

    def test_dataframe_is_cached(self, csv_path):
        loader = DatasetLoader(csv_path)
        first = loader.get_dataframe()
        assert loader.get_dataframe() is first

    def test_missing_file_raises(self, tmp_path):
        loader = DatasetLoader(str(tmp_path / "nope.csv"))
        with pytest.raises(DatasetError):
            loader.get_dataframe()

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(DatasetError):
            DatasetLoader(str(path)).get_dataframe()

    def test_get_rows_returns_schema_objects(self, csv_path, sample_records):
        rows = DatasetLoader(csv_path).get_rows()
        assert len(rows) == len(sample_records)
        assert all(isinstance(row, DatasetRowChurn) for row in rows)


class TestDatasetPreprocessor:
    def setup_method(self):
        self.pre = DatasetPreprocessor()

    def test_split_features_and_target(self, sample_dataframe):
        features, target = self.pre.split_features_and_target(sample_dataframe)
        assert "churn" not in features.columns
        assert target.name == "churn"

    def test_drop_rows_without_target(self, sample_dataframe):
        df = sample_dataframe.copy()
        df.loc[0, "churn"] = None
        result = self.pre.drop_rows_without_target(df)
        assert len(result) == len(sample_dataframe) - 1

    def test_drop_rows_without_target_all_missing(self, sample_dataframe):
        df = sample_dataframe.copy()
        df["churn"] = None
        with pytest.raises(DatasetError):
            self.pre.drop_rows_without_target(df)

    def test_prepare_train_test_split(self, sample_dataframe):
        X_train, X_test, y_train, y_test = self.pre.prepare_train_test_split(
            sample_dataframe
        )
        assert len(X_train) + len(X_test) == len(sample_dataframe)
        assert len(y_train) == len(X_train)

    def test_prepare_train_test_split_empty(self):
        with pytest.raises(DatasetError):
            self.pre.prepare_train_test_split(pd.DataFrame())

    def test_get_feature_types(self):
        types = self.pre.get_feature_types()
        assert "monthly_fee" in types["numeric"]
        assert "region" in types["categorical"]

    def test_get_split_info(self, sample_dataframe):
        _, _, y_train, y_test = self.pre.prepare_train_test_split(sample_dataframe)
        info = self.pre.get_split_info(y_train, y_test)
        assert info["train"]["total"] == len(y_train)
        assert info["test"]["total"] == len(y_test)

    def test_build_preprocessor_fits(self, sample_dataframe):
        preprocessor = self.pre.build_preprocessor()
        features, _ = self.pre.split_features_and_target(sample_dataframe)
        transformed = preprocessor.fit_transform(features)
        assert transformed.shape[0] == len(features)
