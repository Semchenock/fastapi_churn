# FastAPI Churn Service

REST-сервис на FastAPI для предсказания оттока клиентов (customer churn).
Сервис умеет обучать ML-модель на табличном датасете и выдавать предсказания
вероятности оттока по одному или нескольким клиентам.

## Цель сервиса

- Обучить модель классификации оттока на исторических данных о клиентах
  (`POST /model/train`) с выбором алгоритма и гиперпараметров.
- Получать предсказания оттока для новых клиентов (`POST /predict`).
- Хранить артефакт обученной модели, её метаданные и историю обучений,
  чтобы их можно было инспектировать через API.

Поддерживаемые алгоритмы:

- **Logistic Regression** (`logistic_regression`)
- **Random Forest** (`random_forest`)

## Структура проекта

```text
fastapi_churn/
├── main.py                 # точка входа: экспортирует FastAPI-приложение `app`
├── data/
│   └── churn_dataset.csv   # обучающий датасет
├── models/                 # артефакты модели (генерируются при обучении)
│   ├── churn_model.pkl
│   ├── churn_model_metadata.json
│   └── churn_model_history.json
├── src/
│   ├── api/                # FastAPI: приложение, роуты, обработчики ошибок
│   ├── core/               # инфраструктура (логирование)
│   ├── model/              # ML-pipeline (sklearn), сохранение/загрузка модели
│   ├── schemas/            # Pydantic-схемы запросов/ответов и конфигов обучения
│   └── services/           # бизнес-логика (датасет и модель)
└── test/                   # тесты (pytest)
```

## Формат датасета `churn_dataset.csv`

CSV с заголовком; одна строка — один клиент. Целевая колонка — `churn`.

| Колонка              | Тип     | Описание                                              |
|----------------------|---------|-------------------------------------------------------|
| `monthly_fee`        | float   | Месячная плата, > 0                                   |
| `usage_hours`        | float   | Часы использования сервиса, ≥ 0                       |
| `support_requests`   | int     | Число обращений в поддержку, ≥ 0                      |
| `account_age_months` | int     | Возраст аккаунта в месяцах, ≥ 0                       |
| `failed_payments`    | int     | Число неудачных платежей, ≥ 0                         |
| `region`             | string  | Регион: `america`, `europe`, `asia`, `africa`         |
| `device_type`        | string  | Устройство: `desktop`, `mobile`, `tablet`             |
| `payment_method`     | string  | Способ оплаты: `card`, `paypal`, `crypto`             |
| `autopay_enabled`    | int     | Автооплата включена: `0` или `1`                      |
| `churn`              | int     | **Целевая переменная**: `1` — отток, `0` — остался    |

Пример строк:

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

Предобработка (масштабирование числовых признаков, one-hot кодирование
категориальных, заполнение пропусков) выполняется автоматически внутри
sklearn-пайплайна. Данные делятся на train/test в соотношении 80/20
со стратификацией по `churn`.

## Запуск локально

### 1. Установить зависимости

```powershell
pip install -r requirements.txt
```

### 2. Запустить приложение

```powershell
python -m uvicorn main:app --reload
```

### 3. Проверить сервис

```text
http://127.0.0.1:8000/
```

Ожидаемый ответ:

```json
{ "message": "ml churn service is running" }
```

Интерактивная документация (Swagger UI): `http://127.0.0.1:8000/docs`.

## Запуск в Docker

### 1. Собрать образ

```powershell
docker build -t fastapi-churn .
```

### 2. Запустить контейнер

```powershell
docker run --rm -p 8000:8000 fastapi-churn
```

Сервис будет доступен на `http://127.0.0.1:8000/`.

## Эндпоинты

| Метод | Путь               | Назначение                                  |
|-------|--------------------|---------------------------------------------|
| GET   | `/`                | Проверка, что сервис запущен                 |
| GET   | `/health`          | Статус сервиса (модель и датасет)            |
| POST  | `/model/train`     | Обучить модель                               |
| POST  | `/predict`         | Предсказать отток                            |
| GET   | `/model/status`    | Статус обученной модели и метрики            |
| GET   | `/model/schema`    | Схема ожидаемых признаков                    |
| GET   | `/model/metrics`   | История метрик обучений                      |
| GET   | `/dataset/preview` | Превью строк датасета                        |
| GET   | `/dataset/info`    | Размер датасета и распределение `churn`      |
| GET   | `/dataset/split-info` | Информация о train/test разбиении        |

## Примеры запросов

### `POST /model/train`

Минимальный запрос (используются гиперпараметры по умолчанию):

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{ "model_type": "logistic_regression" }'
```

С явными гиперпараметрами:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 10,
      "class_weight": "balanced"
    }
  }'
```

Ответ — метрики на тестовой выборке:

```json
{ "accuracy": 0.81, "f1": 0.55 }
```

### `POST /predict`

Один клиент:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 9.99,
    "usage_hours": 27.92,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 1,
    "region": "america",
    "device_type": "desktop",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

Несколько клиентов — передайте массив объектов того же вида.

Ответ — список предсказаний (по элементу на каждого клиента):

```json
[
  { "churn": 1, "probability": 0.87 }
]
```

- `churn` — предсказанный класс (`1` — отток, `0` — остался);
- `probability` — вероятность предсказанного класса.

> Перед вызовом `/predict` модель должна быть обучена (`/model/train`)
> либо загружена из ранее сохранённого артефакта в `models/`.

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

## Тесты

```powershell
python -m pytest
```
