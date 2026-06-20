from fastapi import APIRouter

from src.schemas.error_response import ErrorResponse
from src.schemas.feature_vector_churn import FeatureVectorChurn
from src.schemas.prediction_response_churn import PredictionResponseChurn
from src.schemas.training_config_churn import TrainingConfigChurn
from src.services.model.model_service import model_service

router = APIRouter()


PREDICT_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Ошибка предсказания (модель не обучена, неверные признаки и т.п.)",
        "content": {
            "application/json": {
                "examples": {
                    "model_not_trained": {
                        "summary": "Модель не обучена",
                        "value": {
                            "code": "model_error",
                            "message": "Model is not trained",
                            "details": None,
                        },
                    },
                    "invalid_feature_count": {
                        "summary": "Неверный набор признаков",
                        "value": {
                            "code": "invalid_feature_count",
                            "message": "Vector #0: invalid feature set",
                            "details": {
                                "index": 0,
                                "missing": ["region"],
                                "extra": ["regn"],
                            },
                        },
                    },
                    "invalid_feature_type": {
                        "summary": "Неверный тип значения признака",
                        "value": {
                            "code": "invalid_feature_type",
                            "message": "Vector #0: field 'monthly_fee' has invalid type",
                            "details": {
                                "index": 0,
                                "field": "monthly_fee",
                                "expected": "float",
                                "got": "str",
                            },
                        },
                    },
                    "prediction_error": {
                        "summary": "Ошибка во время инференса модели",
                        "value": {
                            "code": "prediction_error",
                            "message": "Model failed to produce a prediction",
                            "details": {"reason": "..."},
                        },
                    },
                }
            }
        },
    },
    422: {
        "model": ErrorResponse,
        "description": "Тело запроса не прошло валидацию схемы",
        "content": {
            "application/json": {
                "example": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "type": "greater_than",
                            "loc": ["body", "monthly_fee"],
                            "msg": "Input should be greater than 0",
                        }
                    ],
                }
            }
        },
    },
}


TRAIN_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Ошибка подготовки данных или обучения модели",
        "content": {
            "application/json": {
                "examples": {
                    "dataset_error": {
                        "summary": "Проблема с датасетом",
                        "value": {
                            "code": "dataset_error",
                            "message": "Dataset file not found",
                            "details": None,
                        },
                    },
                    "model_error": {
                        "summary": "Ошибка обучения",
                        "value": {
                            "code": "model_error",
                            "message": "Training failed",
                            "details": None,
                        },
                    },
                }
            }
        },
    },
    422: {
        "model": ErrorResponse,
        "description": "Неверная конфигурация обучения",
        "content": {
            "application/json": {
                "example": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "type": "missing",
                            "loc": ["body", "model_type"],
                            "msg": "Field required",
                        }
                    ],
                }
            }
        },
    },
}


@router.post(
    "/predict",
    response_model=list[PredictionResponseChurn],
    responses=PREDICT_ERROR_RESPONSES,
)
def predict(payload: FeatureVectorChurn | list[FeatureVectorChurn]):
    list_payload: list[FeatureVectorChurn] = payload if isinstance(payload, list) else [payload]
    return model_service.predict(list_payload)


@router.post("/model/train", responses=TRAIN_ERROR_RESPONSES)
async def train_model(payload: TrainingConfigChurn):
    return model_service.train(payload)


@router.get("/model/status")
async def model_status():
    return model_service.status()

@router.get("/model/schema")
async def model_schema():
    return model_service.schema()
