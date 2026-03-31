from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from logistic_regression_model import LogisticRegressionModel

# pip install -r requirements.txt
# uvicorn fastapi_server:app --reload
# pytest test_model.py


app = FastAPI(title="API Server", description="API для логистической регрессии")

# Инициализация модели с параметрами по умолчанию
model = LogisticRegressionModel(b0=42, coefficients=np.array([4.0, 40.0]))


class PredictionRequest(BaseModel):
    """
    Модель запроса для предсказания.
    """
    score_IS: float      # Оценка по ИС
    points_python: float # Баллы по Python


class PredictionResponse(BaseModel):
    """
    Модель ответа с предсказанием.
    """
    prediction: int


@app.get("/")
def read_root():
    """
    Корневой эндпоинт.
    """
    return {"message": "Сервер запущен", "model": "Logistic Regression"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Эндпоинт для получения предсказания от модели логистической регрессии.
    
    Args:
        request: запрос с оценкой по ИС и баллами по Python
        
    Returns:
        PredictionResponse: ответ с предсказанным классом
    """
    try:
        # Нормализация и создание numpy массива
        X = np.array([request.score_IS, request.points_python / 50.0])
        
        # Получение предсказания
        prediction = model.predict(X)
        
        return PredictionResponse(prediction=int(prediction))
    
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


