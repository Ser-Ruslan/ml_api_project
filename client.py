"""Простое консольное клиентское приложение."""
import os
import json
import pickle
import requests
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

HOST = os.getenv("HOST", "127.0.0.1")
PORT = os.getenv("PORT", "8000")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

BASE_URL = f"http://{HOST}:{PORT}"


def _load_local_artifacts() -> tuple[dict, dict]:
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "app" / "models"

    with open(models_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(models_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    return config, label_encoders


def _prepare_features_for_get(payload: dict, config: dict, label_encoders: dict) -> list[float]:
    categorical_cols = config.get(
        "feature_names_categorical",
        ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    )
    numeric_cols = config.get(
        "feature_names_numeric",
        ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
    )
    feature_names = config.get(
        "feature_names",
        numeric_cols + [c + "_encoded" for c in categorical_cols],
    )

    encoded: dict[str, float] = {}
    for col in numeric_cols:
        encoded[col] = float(payload[col])

    for col in categorical_cols:
        le = label_encoders[col]
        encoded[col + "_encoded"] = float(le.transform([str(payload[col])])[0])

    return [float(encoded[name]) for name in feature_names]

def main():
    print("Проверка сервера:")
    print(requests.get(f"{BASE_URL}/ping").json())

    print("\nИнформация о модели:")
    print(requests.get(f"{BASE_URL}{API_PREFIX}/model_info").json())

    payload = {
        "age": 63,
        "sex": "male",
        "cp": "angina",
        "trestbps": 145,
        "chol": 233,
        "fbs": "true",
        "restecg": "hyp",
        "thalach": 150,
        "exang": "fal",
        "oldpeak": 2.3,
        "slope": "down",
        "ca": 0,
        "thal": "fix",
    }

    print("\nPOST-запрос (предсказание):")
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/prediction",
        json=payload
    ).json()
    print("class:", response["class"])
    print("proba:", response["proba"])

    print("\nGET-запрос (предсказание):")
    config, label_encoders = _load_local_artifacts()
    prepared = _prepare_features_for_get(payload, config, label_encoders)
    get_response = requests.get(
        f"{BASE_URL}{API_PREFIX}/prediction",
        params={"features": ",".join(map(str, prepared))},
    ).json()
    print("class:", get_response["class"])
    print("proba:", get_response["proba"])

if __name__ == "__main__":
    main()