from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import mlflow
import uvicorn
import csv
from pathlib import Path

# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/ap1305/MLOps_E2E_1st.mlflow")

# Load model
model_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri="mlflow-artifacts:/edf06d9bb69748a2920e5a0f07d5e011/5b420ca125da4fc7a7bd0c9dac5b7fe5/artifacts/best_model/best_model.pkl"
)
with open(model_local_path, "rb") as f:
    model = pickle.load(f)

# Load scaler
scaler_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri="mlflow-artifacts:/edf06d9bb69748a2920e5a0f07d5e011/57b0d41b9c9d4196ae3989ff2f9d153a/artifacts/scaler_file/scaler.pkl"
)
with open(scaler_local_path, "rb") as f:
    scaler = pickle.load(f)

# FastAPI app
app = FastAPI()

# Input Schema
class WineQualityInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Production log CSV
prod_file = Path("prod_data.csv")
if not prod_file.exists():
    with open(prod_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides",
            "free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol"
        ])

@app.post("/predict")
def predict(data: WineQualityInput):
    X_orig = [
        data.fixed_acidity, data.volatile_acidity, data.citric_acid, data.residual_sugar,
        data.chlorides, data.free_sulfur_dioxide, data.total_sulfur_dioxide, data.density,
        data.pH, data.sulphates, data.alcohol
    ]
    
    # Log production input for drift monitoring
    with open(prod_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(X_orig)
    
    X_scaled = scaler.transform([X_orig])
    pred = model.predict(X_scaled)[0]
    
    return {"predicted_class": int(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
