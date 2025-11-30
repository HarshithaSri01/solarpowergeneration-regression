import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("Solar Power Generation Predictor")

# ---------- Load model and optional scaler safely ----------
@st.cache_data(show_spinner=False)
def load_artifacts():
    model_obj = None
    scaler_obj = None
    model_msg = ""
    # Load model
    try:
        model_obj = joblib.load("random_forest_model.joblib")
        model_msg = "Model loaded successfully."
    except FileNotFoundError:
        model_msg = "Model file not found in repository root: random_forest_model.joblib"
    except Exception as e:
        model_msg = f"Failed to load model: {e}"

    # Load scaler if exists
    if os.path.exists("scaler.pkl"):
        try:
            scaler_obj = joblib.load("scaler.pkl")
        except Exception as e:
            scaler_obj = None
            model_msg += f" | scaler.pkl exists but failed to load: {e}"

    return model_obj, scaler_obj, model_msg

model, scaler, load_message = load_artifacts()

if model is None:
    st.error("Model not loaded: " + load_message)
    st.stop()  # stop further execution so the app doesn't crash on missing model

st.success("Model ready: " + load_message)

# ---------- Determine feature names or count ----------
feature_names = None
n_features = getattr(model, "n_features_in_", None)

if hasattr(model, "feature_names_in_"):
    # sklearn stores original feature names here (if available)
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = None

if feature_names is None and n_features is not None:
    # fallback to generic names if names not available
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

# Final fallback: assume 6 common weather features (keeps backwards compatibility)
if feature_names is None:
    feature_names = ["Temperature", "Humidity", "WindSpeed", "Pressure", "SolarRadiation", "CloudCover"]

# ---------- Build dynamic input UI ----------
st.subheader("Enter Input Features")
inputs = {}
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    col = cols[i % 2]
    # make a friendly label
    label = fname.replace("_", " ").title()
    # Use a numeric input with a reasonable default
    with col:
        inputs[fname] = st.number_input(label, value=0.0, format="%.4f")

# ---------- Prediction block ----------
st.markdown("---")
st.subheader("Prediction")

if st.button("Predict Solar Power"):
    try:
        # Build input array in the model-expected order
        input_array = np.array([[inputs[f] for f in feature_names]])
        # If scaler exists, apply it
        if scaler is not None:
            try:
                input_array = scaler.transform(input_array)
            except Exception as e:
                st.warning("Scaler exists but failed to transform input. Predicting without scaler. Error: " + str(e))

        # Validate shape before predicting
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and input_array.shape[1] != expected:
            st.error(f"Input has {input_array.shape[1]} features but model expects {expected}.")
        else:
            prediction = model.predict(input_array)[0]
            st.success(f"Estimated Solar Power Output: **{prediction:.2f} kW**")
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)

# ---------- Helpful debug info (hidden by default) ----------
with st.expander("Debug info (show if things go wrong)"):
    st.write("Model type:", type(model))
    st.write("n_features_in_:", getattr(model, "n_features_in_", None))
    st.write("feature_names_in_ (if present):", getattr(model, "feature_names_in_", None))
    st.write("scaler loaded:", scaler is not None)
    st.write("feature names used by UI:", feature_names)

 
      
