import os
import subprocess
import boto3

TRAIN_DATA = "/opt/ml/processing/train/train.csv"
VAL_DATA = "/opt/ml/processing/validation/validation.csv"

RESULTS_DIR = "/opt/ml/processing/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = [
    {
        "name": "logistic_regression",
        "script": "train_logistic_regression.py"
    },
    {
        "name": "random_forest",
        "script": "train_random_forest.py"
    },
    {
        "name": "xgboost",
        "script": "train_xgboost.py"
    },
    {
        "name": "neural_net",
        "script": "train_neural_net.py"
    }
]
for model in MODELS:
    model_name = model["name"]
    script = model["script"]

    print(f"\n🚀 Running model: {model_name}")

    output_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)

    os.environ["TRAIN_DATA"] = TRAIN_DATA
    os.environ["VAL_DATA"] = VAL_DATA
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["MODEL_NAME"] = model_name

    subprocess.run(
        ["python", script],
        check=True
    )

print("\nAll models completed successfully.")

s3 = boto3.client("s3")

RESULTS_BUCKET = "ml-code-website"
RESULTS_PREFIX = "results/"

for root, dirs, files in os.walk(RESULTS_DIR):
    for file in files:
        local_path = os.path.join(root, file)
        s3_key = os.path.join(
            RESULTS_PREFIX,
            os.path.relpath(local_path, RESULTS_DIR)
        )

        s3.upload_file(local_path, RESULTS_BUCKET, s3_key)
        print(f"Uploaded: s3://{RESULTS_BUCKET}/{s3_key}")
