import os
import json
import pickle
import pandas as pd


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

    # Create proper dataframe with feature names
    sample = pd.DataFrame(
        [[3, 0, 25, 7.25]],
        columns=["Pclass", "Sex", "Age", "Fare"]
    )

    prediction = model.predict(sample)
    assert prediction is not None