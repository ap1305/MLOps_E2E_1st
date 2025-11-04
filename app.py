from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
model=joblib.load('RandomForestClassifier.pkl')
scaler=joblib.load('scaler.pkl')


app=FastAPI()

class winequality(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_Acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API is running."}

@app.post("/predict")    
def predict(data: winequality):
    X=np.array([[data.fixed_acidity,data.volatile_acidity,data.citric_Acid,data.residual_sugar,data.chlorides,data.free_sulfur_dioxide,data.total_sulfur_dioxide,data.density,data.pH,data.sulphates,data.alcohol]])

    scale_data=scaler.transform(X)
    prediction=model.predict(scale_data)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


