from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
import csv

app = FastAPI()

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request body structure
class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    Fare: float

@app.post("/predict")
def predict(data: Passenger):
    features = np.array([[data.Pclass, data.Sex, data.Age, data.Fare]])

    probs = model.predict_proba(features)
    confidence = float(np.max(probs))
    prediction = int(model.predict(features)[0])

    # Log confidence
    with open("logs/predictions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([confidence])

    return {
        "prediction": prediction,
        "confidence": confidence
    }