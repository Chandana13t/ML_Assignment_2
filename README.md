# OTT Viewer Drop-Off Prediction using Machine Learning

## a. Problem Statement

OTT platforms depend heavily on user engagement. One of the important business challenges is identifying when a viewer is likely to stop watching content. Early drop-off reduces platform engagement and affects subscription retention. If platforms can predict drop-off behaviour early, they can improve content recommendations, content placement, and viewer experience.

The objective of this project is to build machine learning models that can predict whether a viewer will drop off while watching OTT content using viewer behaviour and content-related features.

This is treated as a binary classification problem where: 
- 0 indicates the viewer continues watching 
- 1 indicates the viewer drops off

------------------------------------------------------------------------

## b. Dataset Description

The dataset represents OTT viewer engagement behaviour and contains content level and viewer behaviour level features.

Dataset Summary: 
- Total Records: ~33,000 
- Total Raw Features: 23 
- Final Features Used for Training: 15 
- Target Column: drop_off

Some columns were removed during preprocessing because they were directly related to the target variable or were identifiers. Keeping them could cause data leakage and unrealistic model performance.

Removed Columns: 
- drop_off_probability 
- retention_risk 
- avg_watch_percentage 
- cognitive_load 
- show_id 
- title 
- dataset_version

Categorical features such as platform, genre, dialogue density, and attention requirement were encoded into numerical values before model training.

------------------------------------------------------------------------

## c. Models Used and Evaluation Metrics

The following machine learning models were implemented:

1.  Logistic Regression
2.  Decision Tree
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes
5.  Random Forest
6.  XGBoost

Evaluation Metrics Used: 
- Accuracy 
- AUC Score 
- Precision 
- Recall 
- F1 Score 
- Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Table 1: Model Performance Comparison

  Model                 Accuracy   AUC     Precision   Recall   F1      MCC
  --------------------- ---------- ------- ----------- -------- ------- -------
  Logistic Regression   0.941      0.867   0.815       0.764    0.789   0.755
  Decision Tree         0.942      0.878   0.807       0.787    0.797   0.764
  KNN                   0.929      0.842   0.773       0.719    0.745   0.705
  Naive Bayes           0.625      0.781   0.277       1.000    0.433   0.394
  Random Forest         0.955      0.899   0.863       0.820    0.841   0.815
  XGBoost               0.957      0.907   0.857       0.838    0.848   0.823

------------------------------------------------------------------------

## Table 2: Model Observations

  -----------------------------------------------------------------------
  Model                               Key Observation
  ----------------------------------- -----------------------------------
  Logistic Regression                 Performed well as a baseline model and captured general viewing behaviour patterns.

  Decision Tree                       Captured non-linear relationships and slightly improved recall and F1 score.

  KNN                                 Produced balanced results but slightly lower performance compared to ensemble models.

  Naive Bayes                         Very high recall but low precision due to strong independence assumption between features.

  Random Forest                       Strong overall performance due to ensemble learning and reduced overfitting.

  XGBoost                             Best balance across all metrics due to boosting and sequential error correction.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## d. Streamlit Application

The Streamlit application allows users to: 
- Upload test dataset 
- Select machine learning model 
- View evaluation metrics 
- View confusion matrix 
- View classification report 
- Compare model performance 
- Download prediction results

------------------------------------------------------------------------

## Student Details

Name: Chandana T
BITS ID: 2024DC04048

------------------------------------------------------------------------

## Conclusion

This project demonstrates how machine learning can be used to predict viewer engagement behaviour in OTT platforms. Ensemble models such as Random Forest and XGBoost performed better because they capture complex behavioural patterns. Proper preprocessing and removal of leakage columns played an important role in achieving realistic model performance.
