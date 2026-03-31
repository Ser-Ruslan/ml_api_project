import numpy as np
from logistic_regression_model import LogisticRegressionModel


def test_predict_class_1():
    """
    Тест предсказания класса 1 при больших значениях.
    """
    model = LogisticRegressionModel(b0=42, coefficients=np.array([4.0, 40.0]))
    
    # Входные данные с большими значениями, которые должны дать вероятность > 0.5
    x = np.array([1.0, 40.0])
    # z = 42 + 4*1 + 40*1 = 86
    result = model.predict(x)
    
    assert result == 1
    assert isinstance(result, int)


def test_predict_class_0():
    """
    Тест предсказания класса 0 при малых значениях.
    """
    model = LogisticRegressionModel(b0=42, coefficients=np.array([4.0, 40.0]))
    
    # Входные данные с очень малыми значениями, которые должны дать вероятность < 0.5
    x = np.array([-20.0, -1.0])
    # z = 42 + 4*(-20) + 40*(-1) = 42 - 80 - 40 = -78
    result = model.predict(x)
    
    assert result == 0
    assert isinstance(result, int)


def test_predict_boundary_case():
    """
    Тест пограничного случая, когда вероятность близка к 0.5.
    """
    model = LogisticRegressionModel(b0=0, coefficients=np.array([1.0, 1.0]))
    
    # Подбираем значения так, чтобы z было близко к 0
    x = np.array([0.0, 0.0])
    # z = 0 + 1*0 + 1*0 = 0
    # sigmoid(0) = 0.5, но в нашей реализации 0.5 не > 0.5, поэтому будет 0
    result = model.predict(x)
    
    assert result == 0
    assert isinstance(result, int)


if __name__ == "__main__":
    # Запуск тестов
    test_predict_class_1()
    test_predict_class_0()
    test_predict_boundary_case()
    
    print("Все тесты успешно пройдены!")
