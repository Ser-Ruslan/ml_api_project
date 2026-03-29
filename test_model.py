import numpy as np
import pytest
from logistic_regression_model import LogisticRegressionModel


class TestLogisticRegressionModel:
    """
    Тесты для класса LogisticRegressionModel.
    """
    
    def setup_method(self):
        """
        Настройка перед каждым тестом.
        """
        self.model = LogisticRegressionModel()
    
    def test_predict_class_1(self):
        """
        Тест предсказания класса 1 при больших значениях.
        """
        # Входные данные с большими значениями, которые должны дать вероятность > 0.5
        x = np.array([1.0, 1.0, 1.0])
        # z = 48.6*1 + 2*1 + (45.9/50)*1 = 48.6 + 2 + 0.918 = 51.518
        # sigmoid(51.518) ≈ 1.0, поэтому результат должен быть 1
        result = self.model.predict(x)
        
        assert result == 1
        assert isinstance(result, int)
    
    def test_predict_class_0(self):
        """
        Тест предсказания класса 0 при малых значениях.
        """
        # Входные данные с очень малыми значениями, которые должны дать вероятность < 0.5
        x = np.array([-1.0, -1.0, -1.0])
        # z = 48.6*(-1) + 2*(-1) + (45.9/50)*(-1) = -48.6 - 2 - 0.918 = -51.518
        # sigmoid(-51.518) ≈ 0.0, поэтому результат должен быть 0
        result = self.model.predict(x)
        
        assert result == 0
        assert isinstance(result, int)
    
    def test_predict_boundary_case(self):
        """
        Тест пограничного случая, когда z близко к 0.
        """
        # Подбираем значения так, чтобы z было близко к 0
        # Нужно решить: 48.6*x0 + 2*x1 + 0.918*x2 ≈ 0
        # Возьмем очень маленькие значения
        x = np.array([0.0, 0.0, 0.0])
        # z = 0, sigmoid(0) = 0.5, но в нашей реализации 0.5 не > 0.5, поэтому будет 0
        result = self.model.predict(x)
        
        assert result == 0
        assert isinstance(result, int)
    
    def test_predict_wrong_length(self):
        """
        Тест проверки ошибки при неправильной длине массива.
        """
        # Входные данные с неправильной длиной
        x = np.array([1.0, 2.0])  # Только 2 элемента вместо 3
        
        with pytest.raises(ValueError, match="Входной массив должен содержать ровно 3 элемента"):
            self.model.predict(x)


if __name__ == "__main__":
    # Запуск тестов без pytest
    model = LogisticRegressionModel()
    
    # Тест 1 - класс 1
    x1 = np.array([1.0, 1.0, 1.0])
    result1 = model.predict(x1)
    assert result1 == 1
    print(f"Тест 1 пройден: предсказан класс {result1}")
    
    # Тест 2 - класс 0
    x2 = np.array([-1.0, -1.0, -1.0])
    result2 = model.predict(x2)
    assert result2 == 0
    print(f"Тест 2 пройден: предсказан класс {result2}")
    
    # Тест 3 - пограничный случай
    x3 = np.array([0.0, 0.0, 0.0])
    result3 = model.predict(x3)
    assert result3 == 0
    print(f"Тест 3 пройден: предсказан класс {result3}")
    
    print("Все тесты успешно пройдены!")
