from src.logger import get_logger
from src.exceptions import OTTProjectException


logger = get_logger("trainer", "training.log")


def train_models(models, X_train, y_train):

    try:

        logger.info("Starting model training")

        trained_models = {}

        # train each model

        for name, model in models.items():

            logger.info("Training model: %s", name)

            model.fit(X_train, y_train)

            trained_models[name] = model

            logger.info("Completed training model: %s", name)

        logger.info("All models trained successfully")

        return trained_models

    except Exception as e:
        logger.error("Error during model training")
        raise OTTProjectException(str(e))