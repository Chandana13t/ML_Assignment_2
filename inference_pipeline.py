import pandas as pd

from src.inference import predict
from src.logger import get_logger


logger = get_logger("inference_pipeline", "inference.log")


def main():
    """
    Main entry point for the inference pipeline.
    This function initializes the inference process by logging the start of the pipeline,
    creating an empty DataFrame as sample input, and calling the `predict` function to
    generate predictions based on the sample input. The predictions are then printed to the console.
    Returns:
        None
    """

    logger.info("Starting inference pipeline")

    sample_input = pd.DataFrame()

    preds = predict(sample_input)

    print(preds)


if __name__ == "__main__":
    main()
