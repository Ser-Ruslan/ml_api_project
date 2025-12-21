from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json()["статус"] == "ok"

def test_prediction():
    response = client.post(
        "/api/v1/prediction",
        json={"features": [0, 0, 0, 0, 0, 0]}
    )
    assert response.status_code == 200
    assert "предсказание" in response.json()
