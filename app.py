import streamlit as st
import pandas as pd

from src.inference import predict


st.title("OTT Viewer Drop-off Prediction")

st.write("Upload test dataset to predict viewer drop-off risk.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    input_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:")
    st.dataframe(input_df.head())

    model_choice = st.selectbox(
        "Select Model",
        [
            "LogisticRegression",
            "DecisionTree",
            "KNN",
            "NaiveBayes",
            "RandomForest",
            "XGBoost"
        ]
    )

    if st.button("Run Prediction"):

        preds = predict(input_df, model_choice)

        st.write("Predictions:")
        st.write(preds)
