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
    "matplotlib==3.7.1",
    "xgboost==1.7.6"
])

import matplotlib.pyplot as plt
import xgboost as xgb

# ----------------------------
# 1. Environment variables
# ----------------------------
TRAIN_PATH = os.environ["TRAIN_DATA"]
VAL_PATH = os.environ["VAL_DATA"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
MODEL_NAME = os.environ.get("MODEL_NAME", "xgboost")

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

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ----------------------------
# 3. Train model with eval tracking
# ----------------------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "eta": 0.2
}

evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (dval, "validation")],
    evals_result=evals_result,
    verbose_eval=False
)

# ----------------------------
# 4. Final metric
# ----------------------------
final_auc = evals_result["validation"]["auc"][-1]

metrics = {
    "model_name": "XGBoost",
    "model_key": MODEL_NAME,
    "metric_name": "roc_auc",
    "validation_auc": round(final_auc, 4)
}

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------
# 5. Training curve plot
# ----------------------------
plt.figure()
plt.plot(evals_result["train"]["auc"], label="Train AUC")
plt.plot(evals_result["validation"]["auc"], label="Validation AUC")
plt.xlabel("Boosting Round")
plt.ylabel("AUC")
plt.title("XGBoost Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
plt.close()

print("XGBoost evaluation completed.")
