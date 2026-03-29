import numpy as np
import pytest
from linear_regression_model import LinearRegressionModel


class TestLinearRegressionModel:
    """
    Тесты для класса LinearRegressionModel.
    """
    
    def setup_method(self):
        """
        Настройка перед каждым тестом.
        """
        self.model = LinearRegressionModel()
    
    def test_predict_basic(self):
        """
        Тест базовой функциональности с простыми значениями.
        """
        # Входные данные: [1, 1, 1]
        x = np.array([1.0, 1.0, 1.0])
        expected = 48.6 * 1.0 + 2 * 1.0 + (45.9 / 50) * 1.0
        result = self.model.predict(x)
        
        assert abs(result - expected) < 1e-10
        assert isinstance(result, float)
    
    def test_predict_zero_input(self):
        """
        Тест с нулевыми входными данными.
        """
        # Входные данные: [0, 0, 0]
        x = np.array([0.0, 0.0, 0.0])
        expected = 0.0
        result = self.model.predict(x)
        
        assert result == expected
        assert isinstance(result, float)
    
    def test_predict_specific_values(self):
        """
        Тест с конкретными значениями для проверки вычислений.
        """
        # Входные данные: [2, 3, 50]
        x = np.array([2.0, 3.0, 50.0])
        expected = 48.6 * 2.0 + 2 * 3.0 + (45.9 / 50) * 50.0
        # 48.6 * 2 = 97.2
        # 2 * 3 = 6
        # (45.9 / 50) * 50 = 45.9
        # Итого: 97.2 + 6 + 45.9 = 149.1
        expected = 149.1
        result = self.model.predict(x)
        
        assert abs(result - expected) < 1e-10
        assert isinstance(result, float)
    
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
    model = LinearRegressionModel()
    
    # Тест 1
    x1 = np.array([1.0, 1.0, 1.0])
    result1 = model.predict(x1)
    expected1 = 48.6 * 1.0 + 2 * 1.0 + (45.9 / 50) * 1.0
    assert abs(result1 - expected1) < 1e-10
    print(f"Тест 1 пройден: {result1}")
    
    # Тест 2
    x2 = np.array([0.0, 0.0, 0.0])
    result2 = model.predict(x2)
    assert result2 == 0.0
    print(f"Тест 2 пройден: {result2}")
    
    # Тест 3
    x3 = np.array([2.0, 3.0, 50.0])
    result3 = model.predict(x3)
    expected3 = 149.1
    assert abs(result3 - expected3) < 1e-10
    print(f"Тест 3 пройден: {result3}")
    
    print("Все тесты успешно пройдены!")
