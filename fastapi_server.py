from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from logistic_regression_model import LogisticRegressionModel

# pip install -r requirements.txt
# uvicorn fastapi_server:app --reload
# pytest test_model.py


app = FastAPI(title="API Server", description="API для логистической регрессии")

# Инициализация модели
model = LogisticRegressionModel()


class PredictionRequest(BaseModel):
    """
    Модель запроса для предсказания.
    """
    features: List[float]


class PredictionResponse(BaseModel):
    """
    Модель ответа с предсказанием.
    """
    prediction: float


@app.get("/")
def read_root():
    """
    Корневой эндпоинт.
    """
    return {"message": "Сервер запущен"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Эндпоинт для получения предсказания от модели линейной регрессии.
    
    Args:
        request: запрос с массивом признаков из 3 элементов
        
    Returns:
        PredictionResponse: ответ с предсказанным значением
    """
    try:
        # Преобразование списка в numpy массив
        features_array = np.array(request.features)
        
        # Получение предсказания от модели
        prediction = model.predict(features_array)
        
        return PredictionResponse(prediction=prediction)
    
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))


