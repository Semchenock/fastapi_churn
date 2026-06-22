from src.services.exceptions import AppError
from src.services.dataset.exceptions import DatasetError
from src.services.model.exceptions import ModelError


class TestAppError:
    def test_defaults(self):
        err = AppError("boom")
        assert err.code == "app_error"
        assert err.status_code == 400
        assert err.message == "boom"
        assert err.details is None

    def test_overrides(self):
        err = AppError("bad", code="custom", details={"x": 1}, status_code=422)
        assert err.code == "custom"
        assert err.details == {"x": 1}
        assert err.status_code == 422

    def test_is_exception(self):
        assert isinstance(AppError("x"), Exception)
        assert str(AppError("x")) == "x"


class TestDatasetError:
    def test_default_code(self):
        assert DatasetError("nope").code == "dataset_error"

    def test_is_app_error(self):
        assert isinstance(DatasetError("nope"), AppError)


class TestModelError:
    def test_default_code(self):
        assert ModelError("nope").code == "model_error"

    def test_code_override(self):
        err = ModelError("nope", code="invalid_feature_type")
        assert err.code == "invalid_feature_type"

    def test_is_app_error(self):
        assert isinstance(ModelError("nope"), AppError)
