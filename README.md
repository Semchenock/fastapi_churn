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
