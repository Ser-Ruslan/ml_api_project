"""
СПРАВКА ПО СИНТАКСИСУ ПРОЕКТА
============================

Этот файл объясняет весь синтаксис и структуру проекта.
Проект использует логистическую регрессию для бинарной классификации.
"""

# =============================================================================
# 1. ИМПОРТЫ И ЗАВИСИМОСТИ
# =============================================================================

import numpy as np          # Для работы с массивами и вычислениями
from fastapi import FastAPI # Для создания веб-сервера
from pydantic import BaseModel # Для валидации данных
from typing import List     # Для указания типов
import pytest              # Для написания тестов

# =============================================================================
# 2. КЛАСС МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ - ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# =============================================================================

class LogisticRegressionModel:
    """
    Класс имитации модели логистической регрессии.
    Логистическая регрессия используется для бинарной классификации (0 или 1).
    """
    
    def __init__(self):
        """
        Конструктор - вызывается при создании объекта.
        Устанавливает коэффициенты регрессии.
        """
        self.b0 = 48.6          # Коэффициент для первого признака
        self.b1 = 2             # Коэффициент для второго признака  
        self.b2 = 45.9 / 50     # Коэффициент для третьего признака
    
    def _sigmoid(self, z: float) -> float:
        """
        Сигмоидная функция активации.
        Преобразует любое число в диапазон (0, 1).
        
        Args:
            z: входное значение (линейная комбинация признаков)
            
        Returns:
            float: вероятность в диапазоне (0, 1)
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict(self, x: np.ndarray) -> int:
        """
        Метод предсказания - принимает массив, возвращает класс (0 или 1).
        
        Args:
            x: np.ndarray - numpy массив с 3 элементами
            -> int: возвращает 0 или 1 (класс)
        """
        if len(x) != 3:  # Проверка длины массива
            raise ValueError("Входной массив должен содержать ровно 3 элемента")
        
        # Шаг 1: Вычисление линейной комбинации: z = b0*x0 + b1*x1 + b2*x2
        z = self.b0 * x[0] + self.b1 * x[1] + self.b2 * x[2]
        
        # Шаг 2: Применение сигмоидной функции для получения вероятности
        probability = self._sigmoid(z)
        
        # Шаг 3: Бинарная классификация: если вероятность > 0.5, то класс 1, иначе 0
        return 1 if probability > 0.5 else 0

# =============================================================================
# 3. FASTAPI СЕРВЕР
# =============================================================================

app = FastAPI(title="API Server", description="API для логистической регрессии")

# Инициализация модели
model = LogisticRegressionModel()

# Модели данных для API
class PredictionRequest(BaseModel):
    """
    Модель для входных данных запроса.
    """
    features: List[float]  # Список чисел с плавающей точкой

class PredictionResponse(BaseModel):
    """
    Модель для ответа сервера.
    """
    prediction: int  # Предсказанное значение (0 или 1)

# Эндпоинты (URL адреса)
@app.get("/")
def read_root():
    """
    GET запрос на корневой URL (/)
    Возвращает JSON сообщение
    """
    return {"message": "Сервер запущен"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    POST запрос на /predict
    Принимает JSON с features, возвращает prediction
    """
    try:
        features_array = np.array(request.features)  # Превращаем список в numpy массив
        prediction = model.predict(features_array)   # Получаем предсказание
        return PredictionResponse(prediction=prediction)
    
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

# =============================================================================
# 4. ТЕСТЫ
# =============================================================================

class TestLinearRegressionModel:
    """
    Класс с тестами для проверки работы модели.
    """
    
    def setup_method(self):
        """
        Выполняется перед каждым тестом.
        Создает новый экземпляр модели.
        """
        self.model = LinearRegressionModel()
    
    def test_predict_basic(self):
        """
        Тест базовой функциональности.
        """
        x = np.array([1.0, 1.0, 1.0])  # Входные данные
        expected = 48.6 * 1.0 + 2 * 1.0 + (45.9 / 50) * 1.0  # Ожидаемый результат
        result = self.model.predict(x)  # Получаем результат
        
        # Проверки
        assert abs(result - expected) < 1e-10  # Результат должен быть близок к ожидаемому
        assert isinstance(result, float)        # Результат должен быть числом float
    
    def test_predict_zero_input(self):
        """
        Тест с нулевыми значениями.
        """
        x = np.array([0.0, 0.0, 0.0])
        result = self.model.predict(x)
        assert result == 0.0  # Ноль на входе = ноль на выходе
    
    def test_predict_specific_values(self):
        """
        Тест с конкретными значениями.
        """
        x = np.array([2.0, 3.0, 50.0])
        expected = 149.1  # 48.6*2 + 2*3 + (45.9/50)*50 = 149.1
        result = self.model.predict(x)
        assert abs(result - expected) < 1e-10
    
    def test_predict_wrong_length(self):
        """
        Тест проверки ошибки при неправильной длине массива.
        """
        x = np.array([1.0, 2.0])  # Только 2 элемента вместо 3
        
        # Проверяем, что возникает нужная ошибка
        with pytest.raises(ValueError, match="Входной массив должен содержать ровно 3 элемента"):
            self.model.predict(x)

# =============================================================================
# 5. ОСНОВНЫЕ КОНСТРУКЦИИ PYTHON В ПРОЕКТЕ
# =============================================================================

# Декораторы - @имя_декоратора
@app.get("/")      # Декоратор FastAPI для создания GET эндпоинта
def test_predict_basic():  # Декоратор pytest для обозначения теста

# Типизация - указание типов переменных
def predict(self, x: np.ndarray) -> float:  # x: numpy массив, -> возвращает float
features: List[float]  # features: список чисел float

# Обработка исключений
try:
    # Код который может вызвать ошибку
    result = model.predict(features_array)
except ValueError as e:
    # Обработка ошибки
    raise HTTPException(status_code=400, detail=str(e))

# Проверки (assertions)
assert result == expected      # Проверка равенства
assert isinstance(result, float)  # Проверка типа
assert abs(result - expected) < 1e-10  # Проверка с погрешностью

# =============================================================================
# 6. ЗАПУСК ПРОЕКТА
# =============================================================================

# Команды для запуска:
# pip install -r requirements.txt    # Установка зависимостей
# uvicorn fastapi_server:app --reload  # Запуск сервера
# pytest test_model.py              # Запуск тестов

# URL адреса после запуска:
# http://localhost:8000/             # Корневой эндпоинт
# http://localhost:8000/predict      # Эндпоинт предсказания (POST)
# http://localhost:8000/docs         # Документация Swagger UI

# =============================================================================
# 7. СТРУКТУРА ПРОЕКТА
# =============================================================================

"""
ml_api_project/
├── linear_regression_model.py  # Класс модели МО
├── fastapi_server.py           # FastAPI сервер
├── test_model.py               # Тесты
├── requirements.txt            # Зависимости
└── cheatsheet.py              # Этот файл справки
"""

print("Справка по синтаксису проекта загружена!")
