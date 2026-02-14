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

from sklearn.preprocessing import LabelEncoder

# ---------------- Page Config ----------------
st.set_page_config(page_title="OTT Drop-Off Prediction", layout="wide")

# ---------------- Styling ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.metric-card {
    padding: 10px;
    border-radius: 10px;
    background-color: #f7f7f7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("OTT Viewer Drop-Off Prediction")
st.caption("Machine Learning model evaluation dashboard")

st.divider()

# ---------------- Load Models ----------------
MODEL_DIR = "model"

@st.cache_resource
def load_models():

    model_files = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "Naive_Bayes.pkl",
        "Random Forest": "Random_Forest.pkl",
        "XGBoost": "XGBoost.pkl"
    }

    models = {}

    for name, file in model_files.items():
        path = os.path.join(MODEL_DIR, file)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    return models, scaler


models, scaler = load_models()

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv"]
)

# ---------------- Model Info ----------------
model_info = {
    "Logistic Regression": "Baseline linear classification model.",
    "Decision Tree": "Rule-based tree model capturing decision paths.",
    "KNN": "Instance-based model using nearest neighbours.",
    "Naive Bayes": "Probabilistic model assuming feature independence.",
    "Random Forest": "Ensemble of decision trees for better generalization.",
    "XGBoost": "Boosting based ensemble for strong predictive performance."
}

st.info(f"Selected Model: {selected_model} | {model_info[selected_model]}")

# ---------------- Main ----------------
if uploaded_file:

    with st.spinner("Processing dataset and generating predictions..."):

        df = pd.read_csv(uploaded_file)

        # Dataset summary
        st.subheader("Dataset Overview")

        colA, colB = st.columns(2)
        colA.write(f"Rows: {df.shape[0]}")
        colB.write(f"Columns: {df.shape[1]}")

        with st.expander("Preview Dataset"):
            st.dataframe(df.head())

        if "drop_off" not in df.columns:
            st.error("Dataset must contain drop_off column")
            st.stop()

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

        # Encoding categorical features
        cat_cols = X.select_dtypes(include=["object"]).columns
        encoder = LabelEncoder()

        for col in cat_cols:
            X[col] = encoder.fit_transform(X[col])

        X_scaled = scaler.transform(X)

        model = models[selected_model]

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

    # ---------------- Metrics ----------------
    st.subheader("Model Performance Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    c2.metric("AUC", f"{roc_auc_score(y_true, y_prob):.3f}")
    c3.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
    c5.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
    c6.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")

    st.divider()

    # ---------------- Expandable Results ----------------
    with st.expander("Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

    with st.expander("Classification Report"):
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ---------------- Model Comparison ----------------
    st.subheader("All Model Comparison")

    rows = []

    for name, mdl in models.items():

        yp = mdl.predict(X_scaled)
        yp_prob = mdl.predict_proba(X_scaled)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, yp),
            "AUC": roc_auc_score(y_true, yp_prob),
            "Precision": precision_score(y_true, yp),
            "Recall": recall_score(y_true, yp),
            "F1": f1_score(y_true, yp),
            "MCC": matthews_corrcoef(y_true, yp)
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---------------- Download ----------------
    st.subheader("Export Predictions")

    output = df.copy()
    output["Prediction"] = y_pred
    output["Probability"] = y_prob

    st.download_button(
        "Download Prediction File",
        output.to_csv(index=False),
        file_name="ott_predictions.csv"
    )

else:
    st.warning("Upload a dataset from the sidebar to begin analysis.")
