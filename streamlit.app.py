import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load models
gb = joblib.load("gb_model.pkl")
lr = joblib.load("lr_model.pkl")

# Streamlit Page Config
st.set_page_config(
    page_title="Temperature Forecasting App",
    page_icon="⛅",
    layout="centered"
)

st.title("⛅ Weather Temperature Prediction Web App")
st.write(
    "Enter weather parameters to predict the **mean temperature** using trained "
    "machine learning models (Gradient Boosting and Linear Regression)."
)

# Sidebar model selector
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Gradient Boosting (Best)", "Linear Regression")
)

# Input fields
humidity = st.number_input("Humidity (0–1)", min_value=0.0, max_value=1.0)
pressure = st.number_input("Pressure (kPa)", min_value=0.0)
global_radiation = st.number_input("Global Radiation")
precipitation = st.number_input("Precipitation")
sunshine = st.number_input("Sunshine")
temp_min = st.number_input("Min Temperature (°C)")
temp_max = st.number_input("Max Temperature (°C)")
cloud_cover = st.number_input("Cloud Cover (0–8)", min_value=0, max_value=8)

# Build feature vector in correct order
# IMPORTANT: This order must match the order used during training.
features = np.array([[cloud_cover, humidity, pressure,
                      global_radiation, precipitation,
                      sunshine, temp_min, temp_max]])

# Prediction Button
if st.button("Predict Temperature"):
    if model_choice == "Gradient Boosting (Best)":
        pred = gb.predict(features)[0]
    else:  # Linear Regression
        pred = lr.predict(features)[0]

    st.success(f"Predicted Mean Temperature: {pred:.2f} °C")
