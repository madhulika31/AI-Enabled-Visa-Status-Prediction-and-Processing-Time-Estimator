import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("visa_model.pkl")
country_enc = joblib.load("country_encoder.pkl")
visa_enc = joblib.load("visa_encoder.pkl")
status_enc = joblib.load("status_encoder.pkl")

st.set_page_config(page_title="Visa Status Prediction", layout="centered")

st.title("✈️ AI Visa Status Prediction App")

st.write("Enter details to predict visa approval status")

# Inputs
country = st.selectbox("Select Country", ["USA", "UK", "India", "Canada"])
visa_type = st.selectbox("Visa Type", ["Tourist", "Student"])
documents = st.radio("All documents submitted?", ["Yes", "No"])

doc_value = 1 if documents == "Yes" else 0

# Encode inputs
country_val = country_enc.transform([country])[0]
visa_val = visa_enc.transform([visa_type])[0]

input_data = np.array([[country_val, visa_val, doc_value]])

# Predict
if st.button("Predict Visa Status"):
    prediction = model.predict(input_data)
    result = status_enc.inverse_transform(prediction)[0]

    if result == "Approved":
        st.success("✅ Visa Approved")
    else:
        st.error("❌ Visa Rejected")
