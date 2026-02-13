import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from src.logger import get_logger
from src.exceptions import OTTProjectException


logger = get_logger("evaluator", "evaluation.log")


def evaluate_all_models(models, X_test, y_test):

    try:

        logger.info("Starting evaluation of models")

        results = {}

        for name, model in models.items():

            logger.info("Evaluating model: %s", name)

            preds = model.predict(X_test)

            results[name] = {
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1": f1_score(y_test, preds),
                "AUC": roc_auc_score(y_test, preds),
                "MCC": matthews_corrcoef(y_test, preds)
            }

        logger.info("Evaluation completed")

        return pd.DataFrame(results).T

    except Exception as e:
        logger.error("Error during evaluation")
        raise OTTProjectException(str(e))
