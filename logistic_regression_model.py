import numpy as np


class LogisticRegressionModel:
    """
    Класс для имитации модели машинного обучения логистической регрессии.
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
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Векторизованная сигмоидная функция.
        
        Args:
            z: numpy массив входных значений
            
        Returns:
            np.ndarray: массив значений сигмоиды в диапазоне (0, 1)
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисляет вероятности для входных данных.
        
        Args:
            X: numpy массив признаков shape (n_samples, n_features)
               
        Returns:
            np.ndarray: массив вероятностей shape (n_samples,)
        """
        if X.shape[1] != len(self.B):
            raise ValueError(f"Ожидается {len(self.B)} признаков, получено {X.shape[1]}")
        
        # Векторизованное вычисление линейной комбинации: z = X·B + b0
        z = np.dot(X, self.B) + self.b0
        
        # Применение сигмоидной функции
        return self._sigmoid(z)
    
    def predict(self, x: np.ndarray) -> int:
        """
        Применяет модель логистической регрессии к входным данным.
        
        Args:
            x: numpy массив с признаками [x0, x1, x2, ...]
               
        Returns:
            int: предсказанное значение (0 или 1)
        """
        # Преобразуем в 2D массив для совместимости с predict_proba
        x_reshaped = x.reshape(1, -1) if x.ndim == 1 else x
        
        # Получаем вероятность
        probability = self.predict_proba(x_reshaped)[0]
        
        # Бинарная классификация: если вероятность > 0.5, то класс 1, иначе 0
        return 1 if probability > 0.5 else 0
