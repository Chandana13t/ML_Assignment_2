import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from src.config import MODEL_DIR, SCALER_PATH

st.set_page_config(page_title="OTT Drop Off Prediction", layout="wide")

st.title("OTT Viewer Drop-Off Prediction App")
st.write("Upload test dataset and select model to predict viewer drop-off risk.")

@st.cache_resource
def load_models():

    models = {}

    for file in os.listdir(MODEL_DIR):

        if file.endswith(".pkl") and file != "scaler.pkl":
            model_name = file.replace(".pkl", "")
            models[model_name] = joblib.load(os.path.join(MODEL_DIR, file))

    scaler = joblib.load(SCALER_PATH)

    return models, scaler


models, scaler = load_models()

#  
model_choice = st.selectbox(
    "Select Model",
    list(models.keys())
)


# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)


# PREDICTION LOGIC
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # TARGET SEPARATION
    if "drop_off" not in df.columns:
        st.error("CSV must contain drop_off column for metrics calculation.")
    else:

        y_true = df["drop_off"]

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

        X = df.drop(drop_cols, axis=1, errors="ignore")

        X_scaled = scaler.transform(X)

        model = models[model_choice]

        y_pred = model.predict(X_scaled)

        # METRICS
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("Model Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"],
            "Value": [accuracy, precision, recall, f1, auc, mcc]
        })

        st.dataframe(metrics_df)

        # CONFUSION MATRIX
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

        # CLASSIFICATION REPORT
        st.subheader("Classification Report")

        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
