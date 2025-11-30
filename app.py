# app.py — Block 1: imports + load model + UI title
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("Solar Power Generation Predictor")

# Load the trained model (file should be in the repo root)
@st.cache_data(show_spinner=False)
def load_model():
    model = joblib.load("random_forest_model.joblib")
    return model

# --- Dynamic input generation + safe prediction (replace Blocks 2 & 3) ---
import os

# Determine required feature names / count from model
feature_names = None
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    # fallback to number of features
    n = getattr(model, "n_features_in_", None)
    if n is not None:
        feature_names = [f"feature_{i+1}" for i in range(n)]
    else:
        # final fallback: assume 6 (original manual inputs)
        feature_names = ["Temperature","Humidity","WindSpeed","Pressure","SolarRadiation","CloudCover"]

st.subheader("Enter Input Features")
inputs = {}
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    col = cols[i % 2]
    # Make labels readable
    label = fname.replace("_", " ").title()
    # Use a generous numeric range and a default of 0.0
    with col:
        inputs[fname] = st.number_input(label, value=0.0, format="%.3f")

# Load scaler if present
scaler = None
if os.path.exists("scaler.pkl"):
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.warning("scaler.pkl exists but failed to load: " + str(e))

st.markdown("---")
st.subheader("Prediction")
if st.button("Predict Solar Power"):
    try:
        # Create numpy array in the order model expects
        input_array = np.array([ [ inputs[f] for f in feature_names ] ])
        # If scaler exists, transform
        if scaler is not None:
            input_array = scaler.transform(input_array)
        prediction = model.predict(input_array)[0]
        st.success(f"Estimated Solar Power Output: **{prediction:.2f} kW**")
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)
