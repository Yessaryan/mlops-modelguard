import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("data/titanic.csv")

# Data preprocessing
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data = data.dropna()

# Feature selection
features = ["Pclass", "Sex", "Age", "Fare"]
X = data[features]
y = data["Survived"]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

if accuracy < 0.6:
    raise Exception("Accuracy below threshold!")

# Baseline confidence
probs = model.predict_proba(X_val)
confidences = np.max(probs, axis=1)
baseline_confidence = float(np.mean(confidences))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save baseline confidence
with open("models/baseline_confidence.json", "w") as f:
    json.dump({"baseline_confidence": baseline_confidence}, f, indent=4)

# Save metrics
metrics = {
    "model_name": "LogisticRegression",
    "accuracy": float(accuracy),
    "baseline_confidence": baseline_confidence,
    "confusion_matrix": cm.tolist(),
    "timestamp": datetime.now().isoformat()
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training completed successfully.")