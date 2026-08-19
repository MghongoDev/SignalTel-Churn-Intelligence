"""
Sections 7-9: FastAPI Application with Input Validation and Swagger Documentation
Telco Customer Churn Prediction API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
from pathlib import Path
import os

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts customer churn probability using a trained ML pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Model paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"
model = None
model_metadata = None

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Pydantic request/response models (Section 8: Input Validation)
class CustomerInput(BaseModel):
    gender: str = Field(..., description="Customer gender", example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0/1)")
    Partner: str = Field(..., description="Has partner", example="Yes")
    Dependents: str = Field(..., description="Has dependents", example="No")
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: str = Field(..., description="Has phone service", example="Yes")
    MultipleLines: str = Field(..., description="Multiple lines", example="No")
    InternetService: str = Field(..., description="Internet service type", example="Fiber optic")
    OnlineSecurity: str = Field(..., description="Online security", example="No")
    OnlineBackup: str = Field(..., description="Online backup", example="Yes")
    DeviceProtection: str = Field(..., description="Device protection", example="No")
    TechSupport: str = Field(..., description="Tech support", example="No")
    StreamingTV: str = Field(..., description="Streaming TV", example="Yes")
    StreamingMovies: str = Field(..., description="Streaming movies", example="Yes")
    Contract: str = Field(..., description="Contract type", example="Month-to-month")
    PaperlessBilling: str = Field(..., description="Paperless billing", example="Yes")
    PaymentMethod: str = Field(..., description="Payment method", example="Electronic check")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")

    @validator('*', pre=True)
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Churn prediction (0=No, 1=Yes)")
    churn_label: str = Field(..., description="Human-readable label")
    churn_probability: float = Field(..., description="Probability of churn")

class BatchRequest(BaseModel):
    customers: List[CustomerInput] = Field(..., min_items=1, max_items=100, description="List of customers")

class BatchResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    expected_features: List[str]
    target_labels: Dict[int, str]
    metrics: Dict[str, Any]

@app.on_event("startup")
async def load_model():
    """Load the trained pipeline on startup"""
    global model, model_metadata
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            model_metadata = {
                "model_type": "Logistic Regression",
                "version": "1.0.0",
                "expected_features": [
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges'
                ],
                "target_labels": {0: "No", 1: "Yes"},
                "metrics": {
                    "Accuracy": 0.7402,
                    "Precision": 0.5061,
                    "Recall": 0.7769,
                    "F1": 0.6129,
                    "ROC-AUC": 0.8397
                }
            }
            print("✓ Model loaded successfully")
        else:
            print("⚠ Model file not found. API will return 503 until model is available.")
    except Exception as e:
        print(f"⚠ Failed to load model: {e}")

@app.get("/", tags=["General"])
async def root():
    """Serve the customer churn prediction dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api-info", tags=["General"])
async def api_info():
    """Welcome endpoint with API info and documentation links."""
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API",
        "api_name": "Telco Churn API",
        "version": "1.0.0",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "model_info": "/model-info"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": model_metadata.get("model_type", "Unknown")
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerInput):
    """Single customer churn prediction"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to DataFrame (raw input format expected by pipeline)
        input_df = pd.DataFrame([customer.dict()])
        
        # Make prediction
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        return PredictionResponse(
            prediction=prediction,
            churn_label="Yes" if prediction == 1 else "No",
            churn_probability=round(probability, 4)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(request: BatchRequest):
    """Batch customer churn predictions"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        input_df = pd.DataFrame([c.dict() for c in request.customers])
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            pred_int = int(pred)
            results.append(PredictionResponse(
                prediction=pred_int,
                churn_label="Yes" if pred_int == 1 else "No",
                churn_probability=round(float(prob), 4)
            ))
        
        return BatchResponse(count=len(results), predictions=results)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return model metadata"""
    if model is None or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model information not available"
        )
    
    return ModelInfoResponse(**model_metadata)

# Error handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "errors": exc.errors()}
    )