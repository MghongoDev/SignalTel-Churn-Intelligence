"""
FastAPI service for the Telco Customer Churn Prediction API.
"""

from contextlib import asynccontextmanager
from math import isfinite
from typing import Any, Dict, List, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"
model = None
model_metadata = None

YesNo = Literal["Yes", "No"]
Gender = Literal["Female", "Male"]
ServiceOption = Literal["No", "Yes", "No internet service"]


@asynccontextmanager
async def lifespan(application: FastAPI):
    await load_model()
    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts customer churn probability using a trained ML pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class CustomerInput(BaseModel):
    gender: Gender = Field(..., description="Customer gender", json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0/1)")
    Partner: YesNo = Field(..., description="Has partner", json_schema_extra={"example": "Yes"})
    Dependents: YesNo = Field(..., description="Has dependents", json_schema_extra={"example": "No"})
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: YesNo = Field(..., description="Has phone service", json_schema_extra={"example": "Yes"})
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ..., description="Multiple lines", json_schema_extra={"example": "No"}
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service type", json_schema_extra={"example": "Fiber optic"}
    )
    OnlineSecurity: ServiceOption = Field(..., description="Online security", json_schema_extra={"example": "No"})
    OnlineBackup: ServiceOption = Field(..., description="Online backup", json_schema_extra={"example": "Yes"})
    DeviceProtection: ServiceOption = Field(..., description="Device protection", json_schema_extra={"example": "No"})
    TechSupport: ServiceOption = Field(..., description="Tech support", json_schema_extra={"example": "No"})
    StreamingTV: ServiceOption = Field(..., description="Streaming TV", json_schema_extra={"example": "Yes"})
    StreamingMovies: ServiceOption = Field(..., description="Streaming movies", json_schema_extra={"example": "Yes"})
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract type", json_schema_extra={"example": "Month-to-month"}
    )
    PaperlessBilling: YesNo = Field(..., description="Paperless billing", json_schema_extra={"example": "Yes"})
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ] = Field(..., description="Payment method", json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., gt=0, allow_inf_nan=False, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, allow_inf_nan=False, description="Total charges")

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Churn prediction (0=No, 1=Yes)")
    churn_label: str = Field(..., description="Human-readable label")
    churn_probability: float = Field(..., description="Probability of churn")


class BatchRequest(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=100, description="List of customers")


class BatchResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    expected_features: List[str]
    target_labels: Dict[int, str]
    metrics: Dict[str, Any]


async def load_model():
    """Load the trained pipeline on startup."""
    global model, model_metadata
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_metadata = {
                "model_type": "Logistic Regression",
                "version": "1.0.0",
                "expected_features": [
                    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                    "MonthlyCharges", "TotalCharges",
                ],
                "target_labels": {0: "No", 1: "Yes"},
                "metrics": {
                    "Accuracy": 0.7402,
                    "Precision": 0.5061,
                    "Recall": 0.7769,
                    "F1": 0.6129,
                    "ROC-AUC": 0.8397,
                },
            }
            print("Model loaded successfully")
        else:
            print("Model file not found. API will return 503 until the model is available.")
    except Exception as exc:
        print(f"Failed to load model: {exc}")


def json_safe(value):
    """Replace non-finite floats so validation errors can be sent as JSON."""
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = jsonable_encoder(json_safe(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": "Validation Error", "errors": errors})


@app.get("/", tags=["General"])
async def root():
    """Serve the customer churn prediction dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api-info", tags=["General"])
async def api_info():
    """Return API information and documentation links."""
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API",
        "api_name": "Telco Churn API",
        "version": "1.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "model_info": "/model-info",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Report whether the model is loaded."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": model_metadata.get("model_type", "Unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerInput):
    """Predict churn for one customer."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        return PredictionResponse(
            prediction=prediction,
            churn_label="Yes" if prediction == 1 else "No",
            churn_probability=round(probability, 4),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/predict-batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(request: BatchRequest):
    """Predict churn for up to 100 customers."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([customer.model_dump() for customer in request.customers])
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        results = []
        for prediction, probability in zip(predictions, probabilities):
            prediction_int = int(prediction)
            results.append(
                PredictionResponse(
                    prediction=prediction_int,
                    churn_label="Yes" if prediction_int == 1 else "No",
                    churn_probability=round(float(probability), 4),
                )
            )
        return BatchResponse(count=len(results), predictions=results)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {exc}",
        ) from exc


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return model metadata."""
    if model is None or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model information not available",
        )
    return ModelInfoResponse(**model_metadata)
