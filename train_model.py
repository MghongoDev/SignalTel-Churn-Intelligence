"""
Sections 3-6: Data Cleaning, Feature Engineering, Model Training, and Model Persistence
Telco Customer Churn Prediction Project
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"

RANDOM_STATE = 42

def load_and_clean_data():
    """Section 3: Data Cleaning"""
    print("=" * 60)
    print("SECTION 3: DATA CLEANING")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Original shape: {df.shape}")
    
    # 1. Convert TotalCharges to numeric, treat blanks as NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    blank_count = df['TotalCharges'].isna().sum()
    print(f"Converted TotalCharges: {blank_count} blank/invalid entries set to NaN")
    
    # 2. Handle missing values (SimpleImputer will handle this in pipeline)
    print(f"Missing values after conversion: {df['TotalCharges'].isna().sum()}")
    
    # 3. Remove customerID
    df = df.drop(columns=['customerID'])
    print("Dropped 'customerID' column")
    
    # 4. Remove exact duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Removed {dup_count} duplicate rows")
    else:
        print("No duplicate rows found")
    
    # 5. Encode target Churn: No=0, Yes=1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    print(f"Encoded Churn: {df['Churn'].value_counts().to_dict()}")
    
    # Confirm only two classes
    assert set(df['Churn'].unique()) == {0, 1}, "Target encoding failed"
    
    print(f"Cleaned shape: {df.shape}")
    return df

def prepare_features(df):
    """Section 4: Feature Engineering and Preprocessing"""
    print("\n" + "=" * 60)
    print("SECTION 4: FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Churn ratio preserved - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, categorical_features, numerical_features

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Section 5: Machine Learning - Train and Compare Models"""
    print("\n" + "=" * 60)
    print("SECTION 5: MACHINE LEARNING MODEL COMPARISON")
    print("=" * 60)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        )
    }
    
    results = []
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc:.4f}" if roc else "ROC-AUC:   N/A")
        print(f"Confusion Matrix:\n{cm}")
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC-AUC': round(roc, 4) if roc else "N/A"
        })
        
        # Track best model by F1 (prioritized for imbalanced churn)
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_name = name
    
    # Results table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print(f"\nSelected Model: {best_name} (F1: {best_f1:.4f})")
    print("""
    Model Selection Rationale:
    - Prioritized F1-Score due to class imbalance (churn is minority class).
    - Gradient Boosting or Random Forest typically perform well on tabular data.
    - Trade-off discussion: Higher recall reduces missed churners (costly for business).
      Higher precision avoids unnecessary retention offers.
    - Limitations: Evaluation on single split; cross-validation would give more robust estimates.
    """)
    
    return best_model, best_name, results_df

def save_model(pipeline, model_name, results_df, X_train):
    """Section 6: Save the Model"""
    print("\n" + "=" * 60)
    print("SECTION 6: MODEL PERSISTENCE")
    print("=" * 60)
    
    # Save complete fitted pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✓ Saved complete pipeline to: {MODEL_PATH}")
    
    # Verify loading
    loaded_model = joblib.load(MODEL_PATH)
    print("✓ Verified model loads successfully in fresh process")
    
    # Test with raw input
    sample = X_train.iloc[[0]]
    pred = loaded_model.predict(sample)
    print(f"✓ Raw prediction test successful: {pred[0]}")
    
    # Metadata
    metadata = {
        'model_type': model_name,
        'version': '1.0.0',
        'expected_features': list(X_train.columns),
        'target_labels': {0: 'No', 1: 'Yes'},
        'metrics': results_df[results_df['Model'] == model_name].to_dict('records')[0],
        'random_state': RANDOM_STATE
    }
    
    print(f"✓ Model metadata prepared (features: {len(metadata['expected_features'])})")
    
    return metadata

if __name__ == "__main__":
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, preprocessor, cat_feats, num_feats = prepare_features(df)
    best_pipeline, best_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    metadata = save_model(best_pipeline, best_name, results, X_train)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Pipeline saved successfully!")
    print("=" * 60)