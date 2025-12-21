from pathlib import Path
import json
import joblib
from typing import List
from pydantic import BaseModel

# Базовая директория приложения
BASE_DIR = Path(__file__).resolve().parent
# Директория с моделями
MODELS_DIR = BASE_DIR / "models"

def load_model(model_filename: str | None = None):
    """Загружает ML-модель из папки models.
    Если имя файла не указано — используется первый найденный .pkl/.joblib файл.
    """
    if model_filename is None:
        candidates = list(MODELS_DIR.glob("*.pkl")) + list(MODELS_DIR.glob("*.joblib"))
        if not candidates:
            raise FileNotFoundError("Файл модели не найден")
        model_filename = candidates[0].name

    return joblib.load(MODELS_DIR / model_filename)

def load_model_info(filename: str = "model_info.json") -> dict:
    """Загружает информацию о модели (коэффициенты и R2)."""
    path = MODELS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class Features(BaseModel):
    # Список из 6 числовых признаков
    features: List[float]