import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from .ml_tools import load_model, load_model_info


env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

app = FastAPI(
    title="API линейной регрессии",
    description="Сервис для получения предсказаний линейной регрессии (6 признаков)",
    openapi_url=API_PREFIX + "/openapi.json"
)

# Загружаем модель и информацию о ней при старте сервера
MODEL_FILENAME = os.getenv("MODEL_FILENAME")
model = load_model(MODEL_FILENAME)
model_info = load_model_info()

@app.get("/ping")
def ping():
    """Проверка доступности сервера."""
    return {"статус": "ok"}

class PredictionRequest(BaseModel):
    
    features: List[float]

@app.post(API_PREFIX + "/prediction")
def predict_post(request: PredictionRequest):
    """Получение предсказания (POST)."""
    if len(request.features) != 6:
        raise HTTPException(status_code=400, detail="Ожидается ровно 6 признаков")

    try:
        prediction = model.predict([request.features])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"предсказание": float(prediction)}

@app.get(API_PREFIX + "/prediction")
def predict_get(
    features: Optional[str] = Query(
        None,
        description="6 чисел через запятую, например: 1,2,3,4,5,6"
    )
):
    """Получение предсказания (GET)."""
    if not features:
        raise HTTPException(status_code=400, detail="Параметр features обязателен")

    try:
        values = [float(x) for x in features.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Невозможно преобразовать признаки в числа")

    if len(values) != 6:
        raise HTTPException(status_code=400, detail="Ожидается ровно 6 признаков")

    prediction = model.predict([values])[0]
    return {"предсказание": float(prediction)}

@app.get(API_PREFIX + "/model_info")
def get_model_info():
    """Возвращает коэффициенты модели и значение R2."""
    return JSONResponse(content=model_info)

from app.model_manager import ModelManager

model_manager = ModelManager(
    model_path="app/models/linear_regression_6_features_sklearn_1.7.2.joblib",
    meta_path="app/models/model_info.json"
)


@app.post("/predict_with_meta")
def predict_with_meta(features: list[float]):
    """
    Предсказание с возвратом метаданных модели.
    """
    return {
        "prediction": model_manager.predict(features),
        "r2": model_manager.r2,
        "note": model_manager.note
    }
