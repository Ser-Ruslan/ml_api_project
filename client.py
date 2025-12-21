"""Простое консольное клиентское приложение."""
import os
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

def main():
    print("Проверка сервера:")
    print(requests.get(f"{BASE_URL}/ping").json())

    print("\nИнформация о модели:")
    print(requests.get(f"{BASE_URL}{API_PREFIX}/model_info").json())

    features = [1, 2, 3, 4, 5, 6]

    print("\nPOST-запрос (предсказание):")
    print(
        requests.post(
            f"{BASE_URL}{API_PREFIX}/prediction",
            json={"features": features}
        ).json()
    )

    print("\nGET-запрос (предсказание):")
    print(
        requests.get(
            f"{BASE_URL}{API_PREFIX}/prediction",
            params={"features": ",".join(map(str, features))}
        ).json()
    )

if __name__ == "__main__":
    main()