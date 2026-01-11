import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.server import app

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["статус"] == "ok"

def test_prediction():
    response = client.post(
        "/api/v1/prediction",
        json={
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
    )
    if response.status_code == 500:
        data = response.json()
        assert "detail" in data
        assert "train_model.py" in data["detail"]
        return

    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "proba" in data
