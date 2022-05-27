import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from online_inference.app import app


def test_model_is_not_ready():
    client = TestClient(app)
    response = client.get("/health")
    assert 404 == response.status_code


def test_model_is_ready():
    with TestClient(app) as client:
        response = client.get("/health")
        assert 200 == response.status_code


def test_predict_ok():
    with TestClient(app) as client:
        data = {
            "data": [
                [70, 1, 3, 145, 233, 1, 0, 150, 1, 2.3, 0, 3, 2]
            ],
            "features": [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ],
        }
        response = client.get("/predict/", json=data)
        response_json = response.json()[0]
        assert 200 == response.status_code
        assert "condition" in response_json
        assert "condition_proba" in response_json
        assert 1 == response_json["condition"]
        assert 0.5 <= response_json["condition_proba"]


def test_predict_with_less_features():
    with TestClient(app) as client:
        data = {
            "data": [
                [70, ]
            ],
            "features": [
                "age",
            ],
        }
        response = client.get("/predict/", json=data)
        assert 400 == response.status_code


def test_predict_with_more_features():
    with TestClient(app) as client:
        data = {
            "data": [
                [70, 1, 3, 145, 233, 1, 0, 150, 1, 2.3, 0, 3, 2, 1]
            ],
            "features": [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
                "new_column"
            ],
        }
        response = client.get("/predict/", json=data)
        assert 400 == response.status_code


def test_predict_with_bad_value():
    with TestClient(app) as client:
        data = {
            "data": [
                [1000, 1, 3, 145, 233, 1, 0, 150, 1, 2.3, 0, 3, 2]
            ],
            "features": [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ],
        }
        response = client.get("/predict/", json=data)
        assert 400 == response.status_code
