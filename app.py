import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #123c69, #11998e);
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #e0f2fe;
    font-size: 17px;
}
.metric-card {
    background: #172337;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #293950;
    margin-bottom: 15px;
}
.metric-card h3 {
    color: #94a3b8;
    font-size: 15px;
}
.metric-card h2 {
    color: #5eead4;
    font-size: 27px;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL FILES ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "air_quality_output")

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_air_quality_model.pkl"
)
FEATURES_PATH = os.path.join(
    OUTPUT_DIR, "feature_names.pkl"
)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌍 Air Quality Intelligence</h1>
    <p>Machine learning-powered carbon monoxide (CO) level estimation.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- CHECK MODEL ----------------
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check your air_quality_output folder.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error("Feature names file not found. Check your air_quality_output folder.")
    st.stop()

try:
    model, features = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("About This Project")
st.sidebar.write(
    "This application uses a trained machine learning model "
    "to estimate CO levels from air-quality sensor readings."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("**Algorithm:** Extra Trees Regressor")
st.sidebar.write("**Evaluation:** Chronological split")
st.sidebar.write("**R² score:** 0.835 (83.5%)")

st.sidebar.caption(
    "This score comes from your notebook's chronological evaluation. "
    "It is not a guarantee of real-world accuracy."
)

# ---------------- INPUTS ----------------
st.subheader("Enter Air Quality Readings")
st.write("Enter the seven sensor readings used by your model.")

default_values = {
    "PT08.S1(CO)": 1000.0,
    "C6H6(GT)": 5.0,
    "NOx(GT)": 100.0,
    "NO2(GT)": 80.0,
    "T": 20.0,
    "RH": 50.0,
    "AH": 1.0
}

input_values = {}

left, right = st.columns(2)

for index, feature in enumerate(features):
    column = left if index % 2 == 0 else right

    with column:
        input_values[feature] = st.number_input(
            feature,
            value=float(default_values.get(feature, 0.0)),
            format="%.3f",
            key=f"input_{feature}"
        )

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Predict CO Level", type="primary"):
    input_df = pd.DataFrame(
        [[input_values[feature] for feature in features]],
        columns=features
    )

    try:
        prediction = float(model.predict(input_df)[0])

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Estimated Carbon Monoxide (CO)</h3>
                <h2>{prediction:.3f}</h2>
                <p>Predicted model output — mg/m³, subject to the
                target's original units and preprocessing.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "This is a machine learning estimate, not an official "
            "air-quality measurement or health-safety assessment."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Air Quality Analysis & Prediction | Machine Learning Project"
)