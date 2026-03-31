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
model = LogisticRegressionModel(b0=42, B=np.array([4.0, 40.0]))


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
    probability: float


class BatchPredictionRequest(BaseModel):
    """
    Модель запроса для пакетного предсказания.
    """
    samples: List[List[float]]  # Список выборок признаков


class BatchPredictionResponse(BaseModel):
    """
    Модель ответа для пакетного предсказания.
    """
    predictions: List[int]
    probabilities: List[float]


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
        PredictionResponse: ответ с предсказанным классом и вероятностью
    """
    try:
        # Нормализация и создание numpy массива
        X = np.array([request.score_IS, request.points_python / 50.0])
        
        # Получение вероятности и предсказания
        probability = model.predict_proba(X.reshape(1, -1))[0]
        prediction = model.predict(X)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability)
        )
    
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Эндпоинт для пакетного предсказания.
    
    Args:
        request: запрос со списком выборок признаков
        
    Returns:
        BatchPredictionResponse: ответ с предсказаниями и вероятностями
    """
    try:
        # Преобразование в numpy массив с нормализацией
        X = np.array([
            [sample[0], sample[1] / 50.0] if len(sample) >= 2 
            else [sample[0], 0.0] 
            for sample in request.samples
        ])
        
        # Получение вероятностей и предсказаний
        probabilities = model.predict_proba(X)
        predictions = model.predict_batch(X)
        
        return BatchPredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist()
        )
    
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model_info")
def get_model_info():
    """
    Эндпоинт для получения информации о модели.
    """
    return {
        "model_type": "Logistic Regression",
        "intercept": float(model.b0),
        "coefficients": model.B.tolist(),
        "n_features": len(model.B),
        "feature_names": ["score_IS", "points_python_normalized"]
    }


