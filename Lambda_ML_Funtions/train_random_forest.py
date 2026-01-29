import os
import json
import pandas as pd

import subprocess
import sys

# ----------------------------
# 0. Required as some libraries aren't pre-installed
# ----------------------------

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-cache-dir",
    "numpy==1.23.5",
    "matplotlib==3.7.1"
])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

# ----------------------------
# 1. Environment variables
# ----------------------------
TRAIN_PATH = os.environ["TRAIN_DATA"]
VAL_PATH = os.environ["VAL_DATA"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MODEL_NAME = os.environ.get("MODEL_NAME", "random_forest")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 2. Load data
# ----------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["label"])
y_val = val_df["label"]

# ----------------------------
# 3. Train model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 4. Metrics
# ----------------------------
val_preds = model.predict(X_val)
f1 = f1_score(y_val, val_preds)

metrics = {
    "model_name": "Random Forest",
    "model_key": MODEL_NAME,
    "metric_name": "f1_score",
    "validation_f1": round(f1, 4)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------
# 5. Weights / Feature Importance
# ----------------------------
weights = {
    "type": "random_forest",
    "feature_importance": dict(
        zip(X_train.columns, model.feature_importances_.tolist())
    )
}

with open(os.path.join(OUTPUT_DIR, "weights.json"), "w") as f:
    json.dump(weights, f, indent=2)

# ----------------------------
# 6. Save model artifact
# ----------------------------
joblib.dump(model, os.path.join(OUTPUT_DIR, "model.joblib"))

# ----------------------------
# 7. Feature importance plot
# ----------------------------
importances = model.feature_importances_

plt.figure(figsize=(8, 4))
plt.bar(range(len(importances)), importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()

print("Random Forest evaluation completed successfully.")
