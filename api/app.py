from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained XGBoost pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgboost_model_20250602_140516.pkl')
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Churn Prediction API")

# Define the expected input schema (update fields as per your model)
class CustomerFeatures(BaseModel):
    gender: str = None
    SeniorCitizen: int = None
    Partner: str = None
    Dependents: str = None
    tenure: float = None
    PhoneService: str = None
    MultipleLines: str = None
    InternetService: str = None
    OnlineSecurity: str = None
    OnlineBackup: str = None
    DeviceProtection: str = None
    TechSupport: str = None
    StreamingTV: str = None
    StreamingMovies: str = None
    Contract: str = None
    PaperlessBilling: str = None
    PaymentMethod: str = None
    MonthlyCharges: float = None
    TotalCharges: float = None
    # Add any engineered features if required

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])
    # Preprocess and predict
    try:
        # Use pipeline's preprocess_features and predict methods
        X_processed = pipeline.preprocess_features(input_df, is_training=False)
        pred, proba = pipeline.predict(X_processed)
        # If label encoder exists, decode label
        if pipeline.model_metadata.get('target_encoder'):
            pred_label = pipeline.model_metadata['target_encoder'].inverse_transform(pred)[0]
        else:
            pred_label = int(pred[0])
        return {
            "churn_probability": float(proba[0]),
            "churn_label": pred_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# To run: uvicorn api.app:app --reload
