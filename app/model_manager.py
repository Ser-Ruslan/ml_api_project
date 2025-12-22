
"""
Класс управления ML-моделью.

Инкапсулирует:
- модель
- метрику качества (R2)
- примечание

Метаданные загружаются из отдельного JSON-файла.
"""

import json
import joblib
from typing import Any


class ModelManager:
    """Менеджер ML-модели."""

    def __init__(self, model_path: str, meta_path: str):
        """
        :param model_path: путь к файлу модели
        :param meta_path: путь к JSON-файлу с метаданными
        """
        self.model: Any = self._load_model(model_path)
        self.r2: float
        self.note: str
        self._load_meta(meta_path)

    def _load_model(self, path: str) -> Any:
        """Загружает модель с диска."""
        return joblib.load(path)

    def _load_meta(self, path: str) -> None:
        """Загружает R2 и примечание из JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.r2 = data.get("r2")
        self.note = data.get("note")

    def predict(self, features: list[float]) -> float:
        """Возвращает предсказание модели."""
        return float(self.model.predict([features])[0])
