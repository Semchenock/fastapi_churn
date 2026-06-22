from src.core.logging_config import get_logger
from src.schemas.training_config_churn import TrainingConfigChurn
from src.model.logistic_regression import (
    build_pipeline,
    evaluate_model,
    save_churn_model,
    save_churn_model_metadata,
    train_churn_model,
)
from src.services.dataset.dataset_service import dataset_service

logger = get_logger(__name__)


class TrainingService:
    def train(self, config: TrainingConfigChurn):
        logger.info("Starting model training: model_type=%s", config.model_type)
        try:
            X_train, X_test, y_train, y_test = dataset_service.get_train_test()
            preprocessor = dataset_service.get_processor()

            pipeline = build_pipeline(preprocessor, config)
            trained_model = train_churn_model(pipeline, X_train, y_train)
            metrics = evaluate_model(trained_model, X_test, y_test)
            save_churn_model(trained_model)
            metadata = save_churn_model_metadata(metrics, config)
        except Exception:
            logger.exception("Model training failed: model_type=%s", config.model_type)
            raise

        logger.info(
            "Model training finished: model_type=%s metrics=%s",
            config.model_type,
            metrics,
        )
        return trained_model, metadata


training_service = TrainingService()
