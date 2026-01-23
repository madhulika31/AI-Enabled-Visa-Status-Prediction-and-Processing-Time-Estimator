import streamlit as st
import joblib
import numpy as np
from datetime import datetime, timedelta

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Visa Prediction System",
    page_icon="🌍",
    layout="centered"
)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ------------------ Header ------------------
st.markdown(
    "<h1 style='text-align:center;'>🌍 AI-Enabled Visa Status Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Predict visa approval and estimate processing time using AI</p>",
    unsafe_allow_html=True
)
st.divider()

# ------------------ Sidebar ------------------
st.sidebar.header("📋 Applicant Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=65, value=25)

country = st.sidebar.selectbox(
    "Country of Application",
    ["India", "USA", "UK", "Canada", "Australia"]
)

visa_type = st.sidebar.selectbox(
    "Visa Type",
    ["Student", "Work", "Tourist", "Dependent"]
)

experience = st.sidebar.slider(
    "Work Experience (years)",
    min_value=0, max_value=20, value=2
)

education = st.sidebar.selectbox(
    "Highest Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

# ------------------ Encoding ------------------
country_map = {"India": 0, "USA": 1, "UK": 2, "Canada": 3, "Australia": 4}
visa_map = {"Student": 0, "Work": 1, "Tourist": 2, "Dependent": 3}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

country_enc = country_map[country]
visa_enc = visa_map[visa_type]
edu_enc = edu_map[education]

features = np.array([[age, country_enc, visa_enc, experience, edu_enc]])

# ------------------ Prediction ------------------
st.markdown("### 🔍 Prediction Result")

if st.button("Predict Visa Status 🚀", use_container_width=True):

    prediction = model.predict(features)[0]

    # Optional: if your model supports predict_proba
    try:
        prob = model.predict_proba(features).max() * 100
    except:
        prob = None

    # Processing time logic (demo estimation)
    base_days = 30
    extra_days = visa_enc * 5 + experience
    processing_days = base_days + extra_days

    decision_date = datetime.today() + timedelta(days=processing_days)

    # ----------- Result Card -----------
    if prediction == 1 or str(prediction).lower() == "approved":
        st.success("✅ Prediction: Visa Likely to be APPROVED")
    else:
        st.error("❌ Prediction: Visa May be REJECTED")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⏳ Estimated Processing Days", f"{processing_days} days")

    with col2:
        st.metric("📅 Expected Decision Date", decision_date.strftime("%d %b %Y"))

    if prob:
        st.info(f"📊 Confidence Level: {prob:.2f}%")

# ------------------ Footer ------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by Madhulika | Streamlit AI Project</p>",
    unsafe_allow_html=True
)
