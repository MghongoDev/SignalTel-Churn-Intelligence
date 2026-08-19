# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn for a telecommunications company using scikit-learn pipelines and a production-ready FastAPI service.

## Project Overview

This project builds a complete ML solution for predicting whether a customer will churn. It covers the full lifecycle:

- Data loading and verification
- Exploratory data analysis (EDA) with visualizations
- Data cleaning and preprocessing
- Feature engineering with leakage-safe pipelines
- Model training and comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Model persistence using joblib
- FastAPI REST service with input validation and OpenAPI documentation
- Automated testing with pytest
- Docker containerization

**Final Model**: Logistic Regression (best F1-score of **0.6129**)

---

## Key Findings & Results

### Data Insights
- **Dataset**: 7,021 customers after cleaning (originally 7,043)
- **Churn Rate**: 26.5% (imbalanced — 73.5% No vs 26.5% Yes)
- **Strong Predictors**:
  - **Tenure**: Low-tenure customers (<10 months) churn significantly more
  - **Contract Type**: Month-to-month contracts have ~42% churn vs ~3% for two-year contracts
  - **Internet Service**: Fiber optic customers churn at ~42%
  - **Payment Method**: Electronic check users have ~45% churn rate
  - **Monthly Charges**: Churned customers pay higher median amounts

### Model Performance (Test Set)

| Model                | Accuracy | Precision | Recall | **F1**   | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.7402   | 0.5061    | **0.7769** | **0.6129** | **0.8397** |
| Random Forest        | 0.7744   | 0.6007    | 0.4409 | 0.5085   | 0.8149  |
| Gradient Boosting    | 0.7929   | 0.6401    | 0.4973 | 0.5598   | 0.8351  |

**Selected Model**: Logistic Regression  
**Rationale**: Highest F1-score and best recall — critical for identifying churners early while balancing precision.

---

## Project Structure

```
Telco-Customer-Churn/
├── .gitignore
├── .python-version
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_pipeline.joblib
├── app/
│   └── main.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── notebooks/
│   ├── 01_eda_analysis.py
│   └── eda_outputs/
│       ├── 01_churn_distribution.png
│       ├── 02_tenure_by_churn.png
│       ├── 03_monthly_charges_by_churn.png
│       ├── 04_churn_by_contract.png
│       ├── 05_churn_by_internet.png
│       └── 06_churn_by_payment.png
├── tests/
│   └── test_api.py
├── main.py
├── train_model.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## How to Run the Project

### 1. Setup Environment

```bash
cd Telco-Customer-Churn
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis

```bash
python notebooks/01_eda_analysis.py
```

This generates visualizations in `notebooks/eda_outputs/`.

### 3. Train the Model

```bash
python train_model.py
```

This creates `models/churn_pipeline.joblib`.

### 4. Run the FastAPI Service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Churn risk dashboard: http://localhost:8000/

The dashboard provides a browser-based form for single-customer predictions and
displays the model's churn probability, recommendation, and key performance metrics.

### 5. Run Tests

```bash
pytest -v
```

All 7 tests should pass.

### 6. Docker Deployment

```bash
# Build image
docker build -t churn-api .

# Run container
docker run --rm -p 8000:8000 churn-api
```

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Welcome message + docs links       |
| GET    | `/health`         | Service health & model status      |
| POST   | `/predict`        | Single customer churn prediction   |
| POST   | `/predict-batch`  | Batch predictions (max 100)        |
| GET    | `/model-info`     | Model metadata & performance       |

---

## Example Prediction Request

```json
POST /predict
{
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
```

**Response**:
```json
{
  "prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.7823
}
```

---

## License

This project is for educational purposes.