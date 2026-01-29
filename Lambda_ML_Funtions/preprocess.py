import os
import pandas as pd
from sklearn.model_selection import train_test_split
INPUT_PATH = "/opt/ml/processing/input"
TRAIN_PATH = "/opt/ml/processing/train"
VAL_PATH = "/opt/ml/processing/validation"

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)
files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".csv")]

if len(files) == 0:
    raise ValueError("No CSV file found in input directory")
csv_path = os.path.join(INPUT_PATH, files[0])
print(f"Reading dataset: {csv_path}")

df = pd.read_csv(csv_path)
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.assign(label=y_train).to_csv(
    os.path.join(TRAIN_PATH, "train.csv"), index=False
)
X_val.assign(label=y_val).to_csv(
    os.path.join(VAL_PATH, "validation.csv"), index=False
)
print("Preprocessing completed successfully.")
