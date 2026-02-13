import os
import joblib

from src.config import MODEL_DIR, SCALER_PATH
from src.logger import get_logger


logger = get_logger("inference", "inference.log")


def load_models():

    logger.info("Loading models for inference")

    models = {}

    for file in os.listdir(MODEL_DIR):

        if file.endswith(".pkl") and file != "scaler.pkl":

            model_name = file.replace(".pkl", "")

            models[model_name] = joblib.load(os.path.join(MODEL_DIR, file))

    scaler = joblib.load(SCALER_PATH)

    return models, scaler


def predict(input_df, model_name="RandomForest"):

    logger.info("Running prediction using model: %s", model_name)

    models, scaler = load_models()

    scaled_data = scaler.transform(input_df)

    preds = models[model_name].predict(scaled_data)

    return preds
