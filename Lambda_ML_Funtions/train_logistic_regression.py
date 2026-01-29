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

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------
# 1. Read environment variables
# ----------------------------
TRAIN_PATH = os.environ["TRAIN_DATA"]
VAL_PATH = os.environ["VAL_DATA"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MODEL_NAME = os.environ.get("MODEL_NAME", "logistic_regression")

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
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# 4. Metrics
# ----------------------------
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

train_acc = accuracy_score(y_train, train_preds)
val_acc = accuracy_score(y_val, val_preds)

metrics = {
    "model_name": "Logistic Regression",
    "model_key": MODEL_NAME,
    "metric_name": "accuracy",
    "train_accuracy": round(train_acc, 4),
    "validation_accuracy": round(val_acc, 4)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

weights = {
    "type": "logistic_regression",
    "coefficients": model.coef_[0].tolist(),
    "intercept": model.intercept_.tolist()
}

with open(os.path.join(OUTPUT_DIR, "weights.json"), "w") as f:
    json.dump(weights, f, indent=2)


# ----------------------------
# 5. Accuracy comparison plot
# ----------------------------
plt.figure()
plt.bar(["Train", "Validation"], [train_acc, val_acc])
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.close()

# ----------------------------
# 6. Confusion matrix
# ----------------------------
cm = confusion_matrix(y_val, val_preds)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
