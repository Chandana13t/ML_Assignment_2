import pandas as pd
from src.config import DATA_PATH
from src.logger import get_logger
from src.exceptions import OTTProjectException


logger = get_logger("data_loader", "data_loader.log")


def load_dataset():

    try:
        logger.info("Loading dataset from path: %s", DATA_PATH)

        df = pd.read_csv(DATA_PATH)

        logger.info("Dataset loaded successfully")
        logger.info("Dataset Shape: %s", df.shape)

        return df

    except Exception as e:
        logger.error("Error while loading dataset")
        raise OTTProjectException(str(e))
