import os
import joblib

from src.config import MODEL_DIR, SCALER_PATH, METRICS_PATH
from src.logger import get_logger


logger = get_logger("model_saver", "model_saver.log")


def save_scaler(scaler):

    logger.info("Saving scaler")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)


def save_models(models_dict):

    logger.info("Saving trained models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models_dict.items():

        file_path = os.path.join(MODEL_DIR, f"{name}.pkl")

        joblib.dump(model, file_path)


def save_metrics(results_df):

    logger.info("Saving metrics CSV")

    results_df.to_csv(METRICS_PATH)
