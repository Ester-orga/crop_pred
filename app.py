import streamlit as st
import numpy as np
import pickle
import os

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")

st.title("🌱 Crop Recommendation System")

# -----------------------
# Debug Info (helps avoid blank screen)
# -----------------------
st.write("✅ App started successfully")

# Show files in directory (VERY useful on Streamlit Cloud)
st.write("Files in directory:", os.listdir())

# -----------------------
# Load Model
# -----------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -----------------------
# Inputs
# -----------------------
st.header("Enter Soil Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# -----------------------
# Prediction
# -----------------------
if st.button("🌾 Predict Crop"):

    # Validate input
    if any(v == 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
        st.warning("⚠️ Please enter valid non-zero values for all fields.")
    else:
        try:
            # Prepare input
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Scale input
            data = scaler.transform(data)

            # Predict
            prediction = model.predict(data)

            st.success(f"🌿 Recommended Crop: **{prediction[0]}**")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts the best crop based on soil nutrients and environmental conditions.\n\n"
    "Model: Machine Learning"
)