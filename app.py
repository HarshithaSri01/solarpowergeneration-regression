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

model = load_model()
# Block 2: Input fields
st.subheader("Enter Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=30.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0)

with col2:
    pressure = st.number_input("Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1010.0)
    solar_radiation = st.number_input("Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=500.0)
    cloud_cover = st.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=20.0)
