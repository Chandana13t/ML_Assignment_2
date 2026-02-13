from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.logger import get_logger

logger = get_logger("model_factory", "model_factory.log")


def get_models():

    # Initialize and return a dictionary of classification models
    logger.info("Creating model objects")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    logger.info("Model objects created")

    return models