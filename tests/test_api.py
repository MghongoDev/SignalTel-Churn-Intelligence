"""
Section 10: Automated Testing with pytest and FastAPI TestClient
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, load_model
import json

client = TestClient(app)

# Manually load model for tests
import asyncio
asyncio.run(load_model())

# Valid customer payload for testing
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
    "TotalCharges": 958.20
}

def test_health_check():
    """Test successful health check and model-ready status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_single_success():
    """Test successful single prediction with expected response structure"""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    assert data["prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_type():
    """Test invalid values and incorrect data types"""
    invalid = VALID_CUSTOMER.copy()
    invalid["tenure"] = "twelve"  # wrong type
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_predict_missing_field():
    """Test requests with required fields missing"""
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422

def test_predict_batch_success():
    """Test successful batch predictions, including output order and count"""
    batch_payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
    response = client.post("/predict-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert all("prediction" in p for p in data["predictions"])

def test_predict_batch_empty():
    """Test invalid or empty batch requests"""
    response = client.post("/predict-batch", json={"customers": []})
    assert response.status_code == 422

def test_model_info():
    """Test the model information endpoint and its expected metadata"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "version" in data
    assert "expected_features" in data
    assert "metrics" in data
    assert len(data["expected_features"]) == 19