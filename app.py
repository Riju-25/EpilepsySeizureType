import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load saved model, preprocessor, and label encoder
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🧠 Epilepsy Seizure Type Predictor")

st.markdown("Enter the clinical data below to predict the type of epilepsy seizure.")

# Replace these with actual feature names and types
numerical_features = ['Age', 'EEG_Peak', 'Seizure_Duration', 'Heart_Rate']
categorical_features = ['Gender', 'Medication', 'Seizure_Trigger']

# Create inputs for numerical features
numerical_input = {}
for feature in numerical_features:
    numerical_input[feature] = st.number_input(f"Enter {feature}", step=0.1)

# Create inputs for categorical features
categorical_input = {}
for feature in categorical_features:
    categorical_input[feature] = st.selectbox(f"Select {feature}", ['Yes', 'No', 'Unknown'])

# When the user clicks Predict
if st.button("Predict Seizure Type"):
    # Combine inputs into a DataFrame
    input_data = {**numerical_input, **categorical_input}
    input_df = pd.DataFrame([input_data])

    # Preprocess
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🧾 Predicted Seizure Type: **{predicted_label}**")
