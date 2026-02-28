import json
import pandas as pd
import sys

# Load baseline confidence
try:
    with open("models/baseline_confidence.json") as f:
        baseline = json.load(f)["baseline_confidence"]
except:
    print("Baseline not found.")
    sys.exit(0)

# Load production predictions
try:
    data = pd.read_csv("logs/predictions.csv", header=None)
except:
    print("No predictions logged yet.")
    sys.exit(0)

production_confidence = data[0].mean()

print("Baseline Confidence:", baseline)
print("Production Confidence:", production_confidence)

# Detect degradation
if production_confidence < baseline - 0.15:
    print("MODEL DEGRADATION DETECTED")
    sys.exit(1)
else:
    print("MODEL HEALTHY")