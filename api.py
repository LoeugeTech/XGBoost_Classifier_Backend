import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

# Load the trained model
model = joblib.load("model/model.pkl")

# Define the FastAPI app
app = FastAPI()

# Define input schema
class InputData(BaseModel):
    CreditScore: int
    Geography_Germany: int
    Geography_Spain: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

# Define Prediction endpoint
@app.post("/api/predict")
def predict(data: InputData):
    # Ensure correct feature order and count
    input_features = [
        [
            data.CreditScore,
            data.Geography_Germany,
            data.Geography_Spain,
            data.Gender,
            data.Age,
            data.Tenure,
            data.Balance,
            data.NumOfProducts,
            data.HasCrCard,
            data.IsActiveMember,
            data.EstimatedSalary
        ]
    ]

    # Model prediction
    prediction = model.predict(input_features)
    return {"churn_prediction": int(prediction[0])}

# Endpoint for geography values
@app.get("/api/entities/geography")
def get_geography_labels():
    return {
        "geography": [
            {"value": "Germany", "encoding": [1, 0]},
            {"value": "Spain", "encoding": [0, 1]},
            {"value": "France", "encoding": [0, 0]}
        ]
    }

# Endpoint for gender values
@app.get("/api/entities/gender")
def get_gender_labels():
    return {
        "gender": [
            {"value": 1, "label": "Male"},
            {"value": 0, "label": "Female"}
        ]
    }

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
