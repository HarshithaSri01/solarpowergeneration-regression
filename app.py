import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("🌞 Solar Power Generation Predictor")

# Load model safely
@st.cache_data
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

# If model failed to load, stop the app
if model is None:
    st.error("❌ Model failed to load. Error:")
    st.code(error)
    st.stop()

st.success("✅ Model loaded successfully!")

# ------------------------------------------
# AUTO-DETECT MODEL FEATURES
# ------------------------------------------

# If model has feature names stored
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    # Else fallback → use n_features_in_
    nf = model.n_features_in_
    feature_names = [f"feature_{i+1}" for i in range(nf)]

st.write(f"Model expects **{len(feature_names)} features**")

# ------------------------------------------
# CREATE INPUT BOXES BASED ON MODEL FEATURES
# ------------------------------------------

st.subheader("Enter Input Values")

inputs = {}
cols = st.columns(2)

for i, fname in enumerate(feature_names):
    with cols[i % 2]:
        inputs[fname] = st.number_input(f"{fname}", value=0.0)

# ------------------------------------------
# PREDICT
# ------------------------------------------

st.markdown("---")
if st.button("Predict Solar Power"):
    try:
        # Convert inputs → correct array shape
        input_array = np.array([[inputs[f] for f in feature_names]])

        prediction = model.predict(input_array)[0]

        st.success(f"🔮 Estimated Solar Power Output: **{prediction:.3f} kW**")

    except Exception as e:
        st.error("Prediction failed due to an error:")
        st.code(str(e))

# Debug Info
with st.expander("Debug Info"):
    st.write("Feature names detected:", feature_names)
    st.write("Model type:", type(model))







      
