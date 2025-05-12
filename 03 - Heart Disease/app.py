import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Title
st.title("Heart Disease Prediction System")
st.write("Check your heart health with our AI-powered prediction model")

# Load the pre-trained model, scaler, and feature names
@st.cache_resource
def load_model_and_scaler():
    model = load_model('heart_disease_model.h5')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_scaler()

# Sidebar for input features
st.sidebar.header("Patient Information")

# User Inputs
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Gender", ["Female", "Male"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=1, max_value=300, value=120)
cholesterol = st.sidebar.number_input("Serum Cholesterol (mg/dL)", min_value=1, max_value=600, value=200)
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
resting_ecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.sidebar.number_input("Maximum Heart Rate", min_value=1, max_value=250, value=150)
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.number_input("ST Depression", min_value=-3.0, max_value=10.0, step=0.1, value=1.0)
st_slope = st.sidebar.selectbox("ST Segment Slope", ["Upward", "Flat", "Downward"])

if st.sidebar.button("Check Your Heart Health"):
    input_data = pd.DataFrame({
        'age': [age],
        'resting bp s': [resting_bp],
        'cholesterol': [cholesterol],
        'max heart rate': [max_heart_rate],
        'oldpeak': [oldpeak],
        'sex': [1 if sex == "Male" else 0],
        'chest pain type': [
            1 if chest_pain_type == "Typical Angina" else
            2 if chest_pain_type == "Atypical Angina" else
            3 if chest_pain_type == "Non-Anginal Pain" else 4
        ],
        'fasting blood sugar': [1 if fasting_blood_sugar == "Yes" else 0],
        'resting ecg': [
            0 if resting_ecg == "Normal" else
            1 if resting_ecg == "ST-T Wave Abnormality" else 2
        ],
        'exercise angina': [1 if exercise_angina == "Yes" else 0],
        'ST slope': [
            1 if st_slope == "Upward" else
            2 if st_slope == "Flat" else 3
        ]
    })
    
    
    input_encoded = input_data.reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]
    result = "Potential Heart Disease" if probability > 0.5 else "Healthy Heart"
    confidence = probability * 100 if probability > 0.5 else (1 - probability) * 100

    st.header("Prediction Result")
    if result == "Potential Heart Disease":
        st.error(f"⚠️ {result} (Confidence: {confidence:.2f}%)")
        st.write("Recommendation: Please consult with a healthcare professional.")
    else:
        st.success(f"✅ {result} (Confidence: {confidence:.2f}%)")
        st.write("Recommendation: Maintain a healthy lifestyle and regular check-ups.")

st.sidebar.markdown("---")
st.sidebar.write("AI-powered Heart Health Prediction")
st.sidebar.write("Disclaimer: This is a screening tool, not a definitive diagnosis.")
