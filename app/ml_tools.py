from pathlib import Path
import json
import joblib
import pickle
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
        path = MODELS_DIR / "pipeline.pkl"
        if not path.exists():
            raise FileNotFoundError("Файл модели не найден: ожидается app/models/pipeline.pkl (сначала запусти train_model.py)")
        model_filename = path.name

    path = MODELS_DIR / model_filename
    if path.suffix == ".joblib":
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_info(filename: str = "config.json") -> dict:
    """Загружает информацию о модели (метрики и список признаков)."""
    path = MODELS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_label_encoders(filename: str = "label_encoders.pkl") -> dict:
    """Загружает сохранённые LabelEncoder для категориальных признаков."""
    path = MODELS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)

class Features(BaseModel):
    # Список из 6 числовых признаков
    features: List[float]