import streamlit as st
import pandas as pd
import pickle
import os
from models.preprocess import preprocess_input

st.set_page_config(
    page_title="Visa Processing Time Estimator",
    page_icon="🛂",
    layout="centered"
)

st.title("🛂 AI Enabled Visa Processing Time Estimator")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Inputs
visa_type = st.selectbox("Visa Type", ["Tourist", "Student", "Work"])
country = st.selectbox("Country", ["USA", "UK", "Canada"])
urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])

if st.button("Predict Processing Time"):
    input_data = pd.DataFrame({
        "visa_type": [visa_type],
        "country": [country],
        "urgency": [urgency]
    })

    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)[0]

    st.success(f"Estimated Processing Time: {round(prediction, 2)} days")
