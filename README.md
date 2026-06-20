# FastAPI Churn Service

Minimal FastAPI service for the ML churn project.

## Quick Start

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the app

```powershell
python -m uvicorn main:app --reload
```

### 3. Check the service

Open in a browser:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "message": "ml churn service is running"
}
```

### 4. API docs

FastAPI also provides interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Формат ошибок

Все ошибки возвращаются в едином формате:

```json
{
  "code": "string",
  "message": "string",
  "details": null
}
```

- `code` — машиночитаемый код ошибки;
- `message` — человекочитаемое описание;
- `details` — дополнительный контекст (объект/массив) либо `null`.

### Примеры ошибок `POST /predict`

**Модель не обучена** (`400`):

```json
{
  "code": "model_error",
  "message": "Model is not trained",
  "details": null
}
```

**Неверный набор признаков** (`400`):

```json
{
  "code": "invalid_feature_count",
  "message": "Vector #0: invalid feature set",
  "details": { "index": 0, "missing": ["region"], "extra": ["regn"] }
}
```

**Неверный тип значения признака** (`400`):

```json
{
  "code": "invalid_feature_type",
  "message": "Vector #0: field 'monthly_fee' has invalid type",
  "details": { "index": 0, "field": "monthly_fee", "expected": "float", "got": "str" }
}
```

**Тело запроса не прошло валидацию схемы** (`422`):

```json
{
  "code": "validation_error",
  "message": "Request validation failed",
  "details": [
    { "type": "greater_than", "loc": ["body", "monthly_fee"], "msg": "Input should be greater than 0" }
  ]
}
```

### Примеры ошибок `POST /model/train`

**Проблема с датасетом** (`400`):

```json
{
  "code": "dataset_error",
  "message": "Dataset file not found",
  "details": null
}
```

**Неверная конфигурация обучения** (`422`):

```json
{
  "code": "validation_error",
  "message": "Request validation failed",
  "details": [
    { "type": "missing", "loc": ["body", "model_type"], "msg": "Field required" }
  ]
}
```
