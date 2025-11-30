import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Solar Power Predictor", layout="centered")
st.title("🌞 Solar Power Generation Predictor")

# Load model
@st.cache_data
def load_model():
    try:
        model = joblib.load("model.joblib")   # <-- The new correct model file
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_model()

if model is None:
    st.error("❌ Model failed to load:")
    st.code(error)
    st.stop()

st.success("✅ Model loaded successfully!")

# Auto-detect features
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    n = model.n_features_in_
    feature_names = [f"feature_{i+1}" for i in range(n)]

st.write(f"The model expects **{len(feature_names)}** features.")

st.subheader("Enter Input Values")
inputs = {}
cols = st.columns(2)

for i, fname in enumerate(feature_names):
    with cols[i % 2]:
        inputs[fname] = st.number_input(fname, value=0.0)

st.markdown("---")

if st.button("Predict Solar Power"):
    try:
        arr = np.array([[inputs[f] for f in feature_names]])
        pred = model.predict(arr)[0]
        st.success(f"🔮 Estimated Solar Power Output: **{pred:.3f} kW**")
    except Exception as e:
        st.error("Prediction failed:")
        st.code(str(e))







      
