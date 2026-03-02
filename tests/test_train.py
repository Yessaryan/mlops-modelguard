import os
import json
import pickle

def test_model_file_exists():
    assert os.path.exists("models/model.pkl")

def test_baseline_file_exists():
    assert os.path.exists("models/baseline_confidence.json")

def test_baseline_positive():
    with open("models/baseline_confidence.json") as f:
        data = json.load(f)
    assert data["baseline_confidence"] > 0

def test_model_prediction():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[3, 0, 25, 7.25]])
    assert prediction is not None