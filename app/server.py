import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from .ml_tools import load_label_encoders, load_model, load_model_info


env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

app = FastAPI(
    title="API логистической регрессии",
    description="Сервис для классификации (Heart Disease) на основе логистической регрессии",
    openapi_url=API_PREFIX + "/openapi.json"
)

# Загружаем модель и информацию о ней при старте сервера
MODEL_FILENAME = os.getenv("MODEL_FILENAME")
try:
    # Всегда предпочитаем новый pipeline.pkl (логистическая регрессия).
    # MODEL_FILENAME оставляем только как fallback для ручной отладки.
    model = load_model(None)
except Exception:
    try:
        model = load_model(MODEL_FILENAME)
    except Exception:
        model = None

model_info = load_model_info()
label_encoders = load_label_encoders()

@app.get("/ping")
def ping():
    """Проверка доступности сервера."""
    return {"статус": "ok"}

class PredictionRequest(BaseModel):
    age: float
    sex: str
    cp: str
    trestbps: float
    chol: float
    fbs: str
    restecg: str
    thalach: float
    exang: str
    oldpeak: float
    slope: str
    ca: float
    thal: str


def _encode_request(req: PredictionRequest) -> list[float]:
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена. Сначала запусти train_model.py")
    categorical_cols = model_info.get(
        "feature_names_categorical",
        ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    )
    numeric_cols = model_info.get(
        "feature_names_numeric",
        ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
    )
    feature_names = model_info.get(
        "feature_names",
        numeric_cols + [c + "_encoded" for c in categorical_cols],
    )

    encoded: dict[str, float] = {}
    for col in numeric_cols:
        encoded[col] = float(getattr(req, col))

    for col in categorical_cols:
        le = label_encoders.get(col)
        if le is None:
            raise HTTPException(status_code=500, detail=f"Encoder не найден для признака '{col}'")
        value = str(getattr(req, col))
        try:
            encoded[col + "_encoded"] = float(le.transform([value])[0])
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимое значение '{value}' для признака '{col}': {e}",
            )

    try:
        return [float(encoded[name]) for name in feature_names]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Не найден признак в конфигурации: {e}")


def _to_model_frame(values: list[float]) -> pd.DataFrame:
    feature_names = model_info.get("feature_names")
    if not feature_names:
        raise HTTPException(status_code=500, detail="В config.json отсутствует 'feature_names'")
    if len(values) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Ожидается ровно {len(feature_names)} признаков",
        )
    return pd.DataFrame([values], columns=feature_names)

@app.post(API_PREFIX + "/prediction")
def predict_post(request: PredictionRequest):
    """Получение предсказания (POST)."""
    features = _encode_request(request)
    X = _to_model_frame(features)
    try:
        pred_class = int(model.predict(X)[0])
        pred_proba = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "class": pred_class,
        "proba": pred_proba,
    }

@app.get(API_PREFIX + "/prediction")
def predict_get(
    features: Optional[str] = Query(
        None,
        description="13 чисел через запятую (как в config.feature_names), если хотите отправлять уже подготовленные признаки"
    )
):
    """Получение предсказания (GET)."""
    if not features:
        raise HTTPException(status_code=400, detail="Параметр features обязателен")

    try:
        values = [float(x) for x in features.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Невозможно преобразовать признаки в числа")

    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена. Сначала запусти train_model.py")

    X = _to_model_frame(values)
    pred_class = int(model.predict(X)[0])
    pred_proba = float(model.predict_proba(X)[0][1])
    return {"class": pred_class, "proba": pred_proba}

@app.get(API_PREFIX + "/model_info")
def get_model_info():
    """Возвращает метрики и конфигурацию модели."""
    return JSONResponse(content=model_info)
