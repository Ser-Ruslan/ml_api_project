import numpy as np


class LinearRegressionModel:
    """
    Класс для имитации модели машинного обучения линейной регрессии.
    """
    
    def __init__(self, b0: float = 48.6, coefficients: np.ndarray = None):
        """
        Инициализация модели с заданными коэффициентами регрессии.
        
        Args:
            b0: свободный член 
            coefficients: numpy массив весов признаков
        """
        self.b0 = b0
        
        # Если коэффициенты не переданы, используем значения по умолчанию
        if coefficients is None:
            self.B = np.array([2.0, 45.9 / 50.0])  # b1, b2
        else:
            self.B = np.array(coefficients)
    
    def predict(self, x: np.ndarray) -> float:
        """
        Применяет модель линейной регрессии к входным данным.
        
        Args:
            x: numpy массив с признаками [x0, x1, x2, ...]
               
        Returns:
            float: предсказанное значение
        """
        if len(x) != len(self.B):
            raise ValueError(f"Ожидается {len(self.B)} признаков, получено {len(x)}")
        
        # Вычисление линейной комбинации: y = b0 + b1*x1 + b2*x2 + ...
        result = self.b0 + np.dot(x, self.B)
        
        return float(result)
