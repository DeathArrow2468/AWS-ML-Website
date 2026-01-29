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

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# ----------------------------
# 1. Environment variables
# ----------------------------
TRAIN_PATH = os.environ["TRAIN_DATA"]
VAL_PATH = os.environ["VAL_DATA"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MODEL_NAME = os.environ.get("MODEL_NAME", "neural_net")

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
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 4. Predictions + metric
# ----------------------------
val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)

metrics = {
    "model_name": "Neural Network",
    "model_key": MODEL_NAME,
    "metric_name": "roc_auc",
    "validation_auc": round(auc, 4)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

weights = {
    "type": "neural_network",
    "layers": [
        {
            "layer": i,
            "weights": model.coefs_[i].tolist(),
            "biases": model.intercepts_[i].tolist()
        }
        for i in range(len(model.coefs_))
    ]
}

with open(os.path.join(OUTPUT_DIR, "weights.json"), "w") as f:
    json.dump(weights, f, indent=2)


# ----------------------------
# 5. ROC Curve plot
# ----------------------------
fpr, tpr, _ = roc_curve(y_val, val_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Neural Network ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

print("Neural network evaluation completed.")
