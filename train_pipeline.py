from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.model_factory import get_models
from src.trainer import train_models
from src.evaluator import evaluate_all_models
from src.model_saver import save_models, save_scaler, save_metrics

from src.logger import get_logger


def main():

    logger = get_logger("training_pipeline", "training.log")

    try:

        logger.info("========== TRAINING PIPELINE STARTED ==========")

        # load dataset
        logger.info("Loading dataset")
        df = load_dataset()

        logger.info("Dataset loaded successfully")

        # preprocess data
        logger.info("Starting preprocessing step")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        logger.info("Preprocessing completed")

        # get model objects
        logger.info("Loading model objects")
        models = get_models()

        logger.info("Models loaded successfully")

        # train models
        logger.info("Starting model training")
        trained_models = train_models(models, X_train, y_train)

        logger.info("Model training completed")

        # evaluate models
        logger.info("Starting evaluation")
        results = evaluate_all_models(trained_models, X_test, y_test)

        logger.info("Evaluation completed")
        logger.info("\n%s", results)

        # save artifacts
        logger.info("Saving scaler")
        save_scaler(scaler)

        logger.info("Saving models")
        save_models(trained_models)

        logger.info("Saving metrics")
        save_metrics(results)

        logger.info("========== TRAINING PIPELINE COMPLETED ==========")

        print("\nFinal Model Metrics:")
        print(results)

    except Exception as e:

        logger.exception("Training pipeline failed due to error: %s", str(e))
        raise


if __name__ == "__main__":
    main()
