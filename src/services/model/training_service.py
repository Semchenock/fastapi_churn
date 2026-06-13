from src.model.logistic_regression import (
    build_pipeline,
    evaluate_model,
    save_churn_model,
    save_churn_model_metadata,
    train_churn_model,
)
from src.services.dataset.dataset_service import dataset_service


class TrainingService:
    def train(self):
        X_train, X_test, y_train, y_test = dataset_service.get_train_test()
        preprocessor = dataset_service.get_processor()
        pipeline = build_pipeline(preprocessor)
        trained_model = train_churn_model(pipeline, X_train, y_train)
        metrics = evaluate_model(trained_model, X_test, y_test)
        save_churn_model(trained_model)
        metadata = save_churn_model_metadata(metrics)

        return trained_model, metadata


training_service = TrainingService()
