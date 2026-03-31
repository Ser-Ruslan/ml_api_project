import numpy as np
from logistic_regression_model import LinearRegressionModel


def test_predict_positive():
    """
    Тест предсказания с положительными значениями.
    """
    model = LinearRegressionModel(b0=42, coefficients=np.array([4.0, 40.0]))
    
    # Входные данные
    x = np.array([1.0, 40.0])
    # y = 42 + 4*1 + 40*40 = 42 + 4 + 1600 = 1646
    result = model.predict(x)
    
    assert result == 1646.0
    assert isinstance(result, float)


def test_predict_negative():
    """
    Тест предсказания с отрицательными значениями.
    """
    model = LinearRegressionModel(b0=42, coefficients=np.array([4.0, 40.0]))
    
    # Входные данные
    x = np.array([-10.0, -1.0])
    # y = 42 + 4*(-10) + 40*(-1) = 42 - 40 - 40 = -38
    result = model.predict(x)
    
    assert result == -38.0
    assert isinstance(result, float)


def test_predict_zero():
    """
    Тест предсказания с нулевыми значениями.
    """
    model = LinearRegressionModel(b0=0, coefficients=np.array([1.0, 1.0]))
    
    # Входные данные
    x = np.array([0.0, 0.0])
    # y = 0 + 1*0 + 1*0 = 0
    result = model.predict(x)
    
    assert result == 0.0
    assert isinstance(result, float)


if __name__ == "__main__":
    # Запуск тестов
    test_predict_positive()
    test_predict_negative()
    test_predict_zero()
    
    print("Все тесты успешно пройдены!")
