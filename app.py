import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# Load saved components using cloudpickle
def load_pickle(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)

model = load_pickle("best_model.pkl")
preprocessor = load_pickle("preprocessor.pkl")
label_encoder = load_pickle("label_encoder.pkl")

st.set_page_config(page_title="Epilepsy Seizure Type Predictor", layout="centered")
st.title("🧠 Epilepsy Seizure Type Predictor")
st.markdown("Enter clinical parameters to predict the type of epilepsy seizure.")

# --- Customize these based on your dataset ---
numerical_features = ['Age', 'Weight', 'Height', 'Seizure Frequency', 'Seizure Duration']
categorical_features = [
    'Seizure Type', 'Loss of Consciousness', 'Aura Before Seizure',
    'Jerky Movements', 'Blank Stare Episodes', 'EEG Abnormality Detected',
    'History of Stroke', 'Flashing Lights Sensitivity', 'Family History of Epilepsy',
    'MRI/CT Scan Result', 'Postictal Confusion', 'Lack of Sleep Before Episode'
]

# --- Define input fields ---
user_input = {}

# Numerical Inputs
st.subheader("Numerical Inputs")
for col in numerical_features:
    user_input[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)

# Categorical Inputs
st.subheader("Categorical Inputs")
for col in categorical_features:
    user_input[col] = st.selectbox(f"{col}", ['Yes', 'No'])

# Example extra feature with multiple categories
user_input['Seizure Type'] = st.selectbox("Seizure Type", ['Focal', 'Absence', 'Generalized'])

# --- Predict ---
if st.button("🔍 Predict Seizure Type"):
    input_df = pd.DataFrame([user_input])
    processed_input = preprocessor.transform(input_df)
    prediction = model.predict(processed_input)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"✅ Predicted Seizure Type: **{predicted_label}**")
