import os

BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "ott_viewer_dropoff_retention_us_v1.0.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.csv")

LOG_DIR = os.path.join(BASE_DIR, "logs")
