from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json={"current_delay": 12.5})
    assert response.status_code == 200
    assert response.json() == {"predicted_delay": 12.5}


def test_predict_requires_current_delay():
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_invalid_delay():
    response = client.post("/predict", json={"current_delay": "invalid"})
    assert response.status_code == 422
