import pandas as pd
import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/titanic.csv")

# --- Clean Data Properly ---

# Fill missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())

# Convert Sex column to numeric
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Drop any remaining rows with missing values (safety)
data = data.dropna()

# Select features
features = ["Pclass", "Sex", "Age", "Fare"]
X = data[features]
y = data["Survived"]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

if accuracy < 0.6:
    raise Exception("Accuracy below threshold!")

# Calculate baseline confidence
probs = model.predict_proba(X_val)
confidences = np.max(probs, axis=1)
baseline_confidence = float(np.mean(confidences))

print("Baseline Confidence:", baseline_confidence)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save baseline confidence
with open("models/baseline_confidence.json", "w") as f:
    json.dump({"baseline_confidence": baseline_confidence}, f)

print("Training completed successfully.")