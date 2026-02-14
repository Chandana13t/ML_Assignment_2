# OTT Viewer Drop-off Prediction
# Assignment implementation covering data audit, preprocessing,
# model training, evaluation and artifact saving.


import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# --------------------------
# Paths
# --------------------------
DATA_PATH = "data/ott_viewer_dropoff_retention_us_v1.0.csv"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------
# Data Loading
# --------------------------
def load_dataset():
    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nDataset structure:")
    print(df.info())

    return df


# --------------------------
# Data Quality Checks
# --------------------------
def check_data_quality(df):

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


# --------------------------
# Target Distribution
# --------------------------
def check_target(df):

    print("\nTarget distribution:")
    print(df["drop_off"].value_counts())

    print("\nTarget distribution (%):")
    print(df["drop_off"].value_counts(normalize=True))


# --------------------------
# Correlation Analysis
# --------------------------
def correlation_check(df):

    print("\nCorrelation with drop_off:")
    corr = df.corr(numeric_only=True)
    print(corr["drop_off"].sort_values(ascending=False))


# --------------------------
# Feature Preparation
# --------------------------
def prepare_features(df):

    print("\nPreparing feature matrix...")

    drop_cols = [
        "drop_off",
        "show_id",
        "title",
        "drop_off_probability",
        "dataset_version",
        "avg_watch_percentage",
        "cognitive_load",
        "retention_risk"
    ]

    y = df["drop_off"]
    X = df.drop(drop_cols, axis=1, errors="ignore")

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    for col in cat_cols:
        X[col] = encoder.fit_transform(X[col])

    print("Categorical columns encoded:", list(cat_cols))
    print("Feature matrix shape:", X.shape)

    return X, y


# --------------------------
# Model Definitions
# --------------------------
def build_models():

    print("\nPreparing model list...")

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }


# --------------------------
# Training + Evaluation
# --------------------------
def train_and_evaluate(models, X_train, X_test, y_train, y_test):

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "AUC": roc_auc_score(y_test, preds),
            "MCC": matthews_corrcoef(y_test, preds)
        }

        results.append(metrics)

        # save model
        filename = name.replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, filename))

    return pd.DataFrame(results)


# --------------------------
# Main Execution
# --------------------------
def run_pipeline():

    print("\nStarting training workflow")

    df = load_dataset()

    check_data_quality(df)
    check_target(df)
    correlation_check(df)

    X, y = prepare_features(df)

    print("\nScaling features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("\nSplitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = build_models()

    results_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\nModel comparison results:")
    print(results_df)

    results_df.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)

    print("\nPipeline finished successfully")


# --------------------------
# Run Script
# --------------------------
if __name__ == "__main__":
    run_pipeline()
