import os
import subprocess

TRAIN_DATA = "/opt/ml/processing/train/train.csv"
VAL_DATA = "/opt/ml/processing/validation/validation.csv"

RESULTS_DIR = "/opt/ml/processing/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("CODE DIR CONTENTS:", os.listdir("/opt/ml/processing/code"))
print("TRAIN FILE EXISTS:", os.path.exists(TRAIN_DATA))
print("VAL FILE EXISTS:", os.path.exists(VAL_DATA))

MODELS = [
    ("logistic_regression", "train_logistic_regression.py"),
    ("random_forest", "train_random_forest.py"),
    ("xgboost", "train_xgboost.py"),
    ("neural_net", "train_neural_net.py"),
]

for name, script in MODELS:
    print(f"Running model: {name}")

    output_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    os.environ["TRAIN_DATA"] = TRAIN_DATA
    os.environ["VAL_DATA"] = VAL_DATA
    os.environ["OUTPUT_DIR"] = output_dir
    os.environ["MODEL_NAME"] = name

    script_path = os.path.join("/opt/ml/processing/code", script)

    try:
        subprocess.run(
            ["python", script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"{name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"{name} failed")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

print("RESULTS DIR TREE:")
for root, dirs, files in os.walk("/opt/ml/processing/results"):
    print(root, dirs, files)

print("All models completed successfully.")
