import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load saved model files
# -----------------------------
model = pickle.load(open("model_files/ridge_model.pkl", "rb"))
scaler = pickle.load(open("model_files/scaler.pkl", "rb"))
columns = pickle.load(open("model_files/columns.pkl", "rb"))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Predictor")
st.write("Enter house details to estimate the price (in Lakhs)")

# -----------------------------
# User inputs
# -----------------------------
total_sqft = st.number_input("Total Square Feet", min_value=300, step=50)
bhk = st.number_input("BHK", min_value=1, max_value=20, step=1)
bath = st.number_input("Bathrooms", min_value=1, max_value=20, step=1)
balcony = st.number_input("Balconies", min_value=0, max_value=10, step=1)

location_list = sorted(
    [c.replace("location_", "") for c in columns if c.startswith("location_")]
)

location = st.selectbox("Location", location_list)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    # Create empty input dataframe
    input_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Fill numeric values
    input_df["total_sqft"] = total_sqft
    input_df["bhk"] = bhk
    input_df["bath"] = bath
    input_df["balcony"] = balcony

    # Set location
    loc_col = "location_" + location
    if loc_col in input_df.columns:
        input_df[loc_col] = 1

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Estimated Price: ₹ {prediction:.2f} Lakhs")
