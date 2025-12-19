# api/app.py
"""
FastAPI service for Bank Customer Churn Prediction.
Loads the trained Optuna-tuned model and exposes a /predict endpoint.
"""

import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import churning_pipeline

sys.modules['housing_pipeline'] = churning_pipeline

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# 1. Import shared pipeline components
# -----------------------------------------------------------------------------

from churning_pipeline import (
    column_ratio,
    ratio_name,
    build_preprocessing,
    make_estimator_for_name,
)

# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = Path("models/global_best_model_optuna.pkl")

app = FastAPI(
    title="Bank Churn Prediction API",
    description="FastAPI service for predicting customer churn probability",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# 3. Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    """Load the trained model from disk."""
    if not path.exists():
        path = Path("../models/global_best_model_optuna.pkl")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

    print(f"Loading model from: {path}")
    m = joblib.load(path)
    print("✓ Model loaded successfully!")
    print(f"  Model type: {type(m).__name__}")
    return m

model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("\n" + "=" * 80)
    print("Churn Prediction API - Starting Up")
    print("=" * 80)
    try:
        model = load_model(MODEL_PATH)
        print("API is ready to accept requests!")
    except Exception as e:
        print(f"✗ ERROR: Failed to load model. {e}")
    print("=" * 80 + "\n")

# -----------------------------------------------------------------------------
# 4. Request / Response Schemas (Data Models)
# -----------------------------------------------------------------------------

class CustomerInstance(BaseModel):
    """
    Define the input features for a single customer instance.
    """
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float

    class Config:
        schema_extra = {
            "example": {
                "credit_score": 600,
                "geography": "France",
                "gender": "Male",
                "age": 40,
                "tenure": 3,
                "balance": 60000.0,
                "num_of_products": 2,
                "has_cr_card": 1,
                "is_active_member": 1,
                "estimated_salary": 50000.0
            }
        }

class PredictRequest(BaseModel):
    """
    Schema for prediction request containing multiple customer instances.
    """
    instances: List[CustomerInstance]

class PredictResponse(BaseModel):
    """
    Return schema for prediction response.
    """
    predictions: List[int]          # 0 or 1
    probabilities: List[float]      # churn probability (0.0 - 1.0)
    churn_status: List[str]         # "Churn" or "Stay"
    count: int

# -----------------------------------------------------------------------------
# 5. Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Bank Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    try:
        # 1. transform input to DataFrame
        data_list = [item.dict() for item in request.instances]
        X = pd.DataFrame(data_list)
        
        # 2. dummy columns to satisfy pipeline requirements
        X["row_number"] = 1
        X["customer_id"] = 10000000
        X["surname"] = "API_User"
        X["exited"] = 0 # Dummy target
        
        # 3. lowercase columns for consistency
        X.columns = X.columns.str.lower()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format. Could not convert to DataFrame: {e}",
        )

    # 4. Check required columns
    required_columns = [
        "credit_score", "geography", "gender", "age", "tenure", 
        "balance", "num_of_products", "has_cr_card", "is_active_member", "estimated_salary"
    ]
    missing = set(required_columns) - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        # 5. Make predictions
        preds = model.predict(X) # 0 or 1
        
        # 6. Get probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = [float(p) for p in preds]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    # 7. Output formatting
    preds_list = [int(p) for p in preds]
    probs_list = [float(p) for p in probs]
    status_list = ["Churn (High Risk)" if p == 1 else "Stay (Low Risk)" for p in preds_list]

    return PredictResponse(
        predictions=preds_list,
        probabilities=probs_list,
        churn_status=status_list,
        count=len(preds_list)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)