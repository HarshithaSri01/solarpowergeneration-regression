import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import traceback

st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("Solar Power Generation Predictor")

# ---------- Debugging load_artifacts() ----------
@st.cache_data(show_spinner=False)
def load_artifacts():
    model_obj = None
    scaler_obj = None
    model_msg = ""
    try:
        # list files in working dir (repo root) with sizes
        files = []
        for entry in sorted(os.listdir(".")):
            try:
                size = os.path.getsize(entry)
            except Exception:
                size = None
            files.append((entry, size))
        # build a readable file list string
        file_parts = []
        for n, s in files:
            if s is None:
                file_parts.append(f"{n}(?)")
            else:
                # show size in KB or MB
                if s < 1024:
                    file_parts.append(f"{n}({s}B)")
                elif s < 1024**2:
                    file_parts.append(f"{n}({s/1024:.1f}KB)")
                else:
                    file_parts.append(f"{n}({s/1024**2:.1f}MB)")
        model_msg += "Repo files: " + ", ".join(file_parts) + " || "

        # attempt to load model and capture full exception details if any
        try:
            model_obj = joblib.load("random_forest_model.joblib")
            model_msg += "Model loaded successfully."
        except Exception as e_model:
            tb = traceback.format_exc()
            model_msg += "Failed to load model: " + repr(e_model) + " | traceback: " + tb

        # load scaler if exists
        if os.path.exists("scaler.pkl"):
            try:
                scaler_obj = joblib.load("scaler.pkl")
                model_msg += " | Scaler loaded."
            except Exception as e_scaler:
                tb2 = traceback.format_exc()
                model_msg += " | scaler.pkl exists but failed: " + repr(e_scaler) + " | traceback: " + tb2
    except Exception as e:
        tb_top = traceback.format_exc()
        model_msg += "Top-level debug failure: " + repr(e) + " | traceback: " + tb_top

    return model_obj, scaler_obj, model_msg

# Load artifacts
model, scaler, load_message = load_artifacts()

if model is None:
    st.error("Model not loaded: " + load_message)
    # show debug info in expander as well
    with st.expander("Full debug output"):
        st.write(load_message)
    st.stop()

st.success("Model ready: " + ("Model loaded successfully." if "Model loaded successfully." in load_message else load_message))

# ---------- Determine feature names or count ----------
feature_names = None
n_features = getattr(model, "n_features_in_", None)

if hasattr(model, "feature_names_in_"):
    try:
        feature_names = list(model.feature_names_in_)
    except Exception:
        feature_names = None

if feature_names is None and n_features is not None:
    feature_names = [f"feature_{i+1}" for i in range(n_features)]

# Final fallback
if feature_names is None:
    feature_names = ["Temperature", "Humidity", "WindSpeed", "Pressure", "SolarRadiation", "CloudCover"]

# ---------- Build dynamic input UI ----------
st.subheader("Enter Input Features")
inputs = {}
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    col = cols[i % 2]
    label = fname.replace("_", " ").title()
    with col:
        # try to set sensible defaults for common names
        default = 0.0
        if "temp" in fname.lower():
            default = 30.0
        elif "humid" in fname.lower():
            default = 50.0
        elif "solar" in fname.lower() or "radiat" in fname.lower():
            default = 500.0
        elif "cloud" in fname.lower():
            default = 20.0
        elif "wind" in fname.lower():
            default = 5.0
        elif "press" in fname.lower():
            default = 1010.0
        inputs[fname] = st.number_input(label, value=float(default), format="%.4f")

# ---------- Prediction block ----------
st.markdown("---")
st.subheader("Prediction")

if st.button("Predict Solar Power"):
    try:
        input_array = np.array([[inputs[f] for f in feature_names]])
        if scaler is not None:
            try:
                input_array = scaler.transform(input_array)
            except Exception as e:
                st.warning("Scaler exists but failed to transform input. Predicting without scaler. Error: " + str(e))

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
    st.write("Working directory:", os.getcwd())
    st.write("Repo files & sizes shown above in message.")
    st.write("Model type:", type(model))
    st.write("n_features_in_:", getattr(model, "n_features_in_", None))
    st.write("feature_names_in_ (if present):", getattr(model, "feature_names_in_", None))
    st.write("scaler loaded:", scaler is not None)
    st.write("feature names used by UI:", feature_names)
    st.write("Load message (raw):")
    st.write(load_message)









      
