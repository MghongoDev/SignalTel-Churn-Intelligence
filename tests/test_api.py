"""
Automated tests for the FastAPI service.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_success(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    assert data["prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1


def test_predict_invalid_type(client):
    invalid = VALID_CUSTOMER.copy()
    invalid["tenure"] = "twelve"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["detail"] == "Validation Error"
    assert response.json()["errors"][0]["loc"] == ["body", "tenure"]


def test_predict_missing_field(client):
    incomplete = {key: value for key, value in VALID_CUSTOMER.items() if key != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert response.json()["detail"] == "Validation Error"


def test_predict_unknown_category(client):
    invalid = VALID_CUSTOMER.copy()
    invalid["Contract"] = "banana"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["errors"][0]["type"] == "literal_error"


def test_predict_nan_charge(client):
    invalid = VALID_CUSTOMER.copy()
    invalid["MonthlyCharges"] = float("nan")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["detail"] == "Validation Error"
    assert response.json()["errors"][0]["type"] == "finite_number"


def test_predict_infinite_total_charges(client):
    invalid = VALID_CUSTOMER.copy()
    invalid["TotalCharges"] = float("inf")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert response.json()["errors"][0]["type"] == "finite_number"


def test_predict_batch_success(client):
    batch_payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
    response = client.post("/predict-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert all("prediction" in prediction for prediction in data["predictions"])


def test_predict_batch_empty(client):
    response = client.post("/predict-batch", json={"customers": []})
    assert response.status_code == 422
    assert response.json()["detail"] == "Validation Error"


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "version" in data
    assert "expected_features" in data
    assert "metrics" in data
    assert len(data["expected_features"]) == 19


def test_model_info_comes_from_artifact(client):
    metadata_path = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline.metadata.json"
    expected = json.loads(metadata_path.read_text())
    data = client.get("/model-info").json()
    assert data["model_type"] == expected["model_type"]
    assert data["metrics"] == expected["metrics"]
    assert data["target_labels"] == expected["target_labels"]
