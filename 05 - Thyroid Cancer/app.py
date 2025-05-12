import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# Set page title and layout
st.set_page_config(page_title="Thyroid Cancer Recurrence Prediction", layout="wide")

# Title and description
st.title("Thyroid Cancer Recurrence Prediction")
st.markdown("""
This application predicts the likelihood of thyroid cancer recurrence based on patient data.
Enter the patient information below to get a prediction.
""")

# Function to load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        # Load the model
        model = load_model('thyroid_model.h5')
        
        # Load the preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
            
        # Load the optimal threshold
        with open('threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
            
        return model, preprocessor, threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # If model files don't exist, provide a message
        st.error("Model files not found. Please run the thyroid_cancer_model.ipynb notebook first to train and save the model.")
        return None, None, None

# Load model and preprocessor
model, preprocessor, threshold = load_model_and_preprocessor()

# Function to get feature values from the dataset
@st.cache_data
def get_feature_values():
    # This is a simplified version - in a real app, you would load these from the saved model
    feature_values = {
        'Gender': ['M', 'F'],
        'Smoking': ['Yes', 'No'],
        'Hx Smoking': ['Yes', 'No'],
        'Hx Radiothreapy': ['Yes', 'No'],
        'Thyroid Function': ['Euthyroid', 'Hypothyroid', 'Hyperthyroid'],
        'Physical Examination': ['Multinodular goiter', 'Normal', 'Nodule'],
        'Adenopathy': ['Bilateral', 'No', 'Extensive', 'Left', 'Right', 'Posterior','Unilateral'],
        'Pathology': ['Papillary', 'Follicular', 'Medullary', 'Anaplastic'],
        'Focality': ['Multi-Focal', 'Uni-Focal'],
        'Risk': ['High', 'Intermediate', 'Low'],
        'T': ['T1a', 'T1b', 'T2', 'T3', 'T4a', 'T4b'],
        'N': ['N0', 'N1a', 'N1b'],
        'M': ['M0', 'M1'],
        'Stage': ['I', 'II', 'III', 'IVA', 'IVB', 'IVC'],
        'Response': ['Excellent', 'Biochemical Incomplete', 'Structural Incomplete', 'Indeterminate']
    }
    return feature_values

feature_values = get_feature_values()

# Create sidebar for inputs
st.sidebar.header("Patient Information")

# Function to create input fields
def create_input_fields():
    inputs = {}
    
    # Age input (numerical)
    inputs['Age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=50)
    
    # Categorical inputs
    for col, values in feature_values.items():
        inputs[col] = st.sidebar.selectbox(f"{col}", options=values)
    
    return inputs

# Get user inputs
user_inputs = create_input_fields()

# Create a dataframe from user inputs
input_df = pd.DataFrame([user_inputs])

# Button to make prediction
if st.sidebar.button("Predict Recurrence"):
    if model is not None and preprocessor is not None and threshold is not None:
        # Preprocess the input data
        input_processed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction_prob = model.predict(input_processed)[0][0]
        prediction = "Yes" if prediction_prob > threshold else "No"
        
        # Display results
        st.header("Prediction Results")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            for key, value in user_inputs.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("Recurrence Prediction")
            st.markdown(f"""
            <div style="background-color:{'#1e3d58' if prediction == 'Yes' else '#526a40'}; padding:20px; border-radius:10px;">
                <h3>Predicted Recurrence: {prediction}</h3>
                <h4>Probability: {prediction_prob:.2%}</h4>
                <p>Threshold: {threshold:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk interpretation
            st.subheader("Risk Interpretation")
            if prediction_prob < threshold * 0.5:
                st.write("**Low risk** of cancer recurrence.")
            elif prediction_prob < threshold:
                st.write("**Moderate risk** of cancer recurrence.")
            else:
                st.write("**High risk** of cancer recurrence.")
            
            st.write("**Note:** This prediction is based on a machine learning model and should be used as a supportive tool for medical professionals, not as a definitive diagnosis.")
    else:
        st.error("Model not loaded. Please run the thyroid_cancer_model.ipynb notebook first to train and save the model.")

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This prediction model uses a deep neural network to predict thyroid cancer recurrence based on patient data.
    
    **Features used in the model:**
    - Patient demographics (Age, Gender)
    - Medical history (Smoking, Radiotherapy history)
    - Clinical findings (Thyroid Function, Physical Examination)
    - Cancer characteristics (Pathology, Focality, Risk, TNM staging)
    
    The model was trained on historical patient data with special techniques to handle class imbalance:
    - SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples of the minority class
    - Class weighting to give more importance to the minority class
    - Optimized threshold selection to improve prediction of the "Yes" class
    """)


with st.expander("About Thyroid Cancer"):
    st.write("""
    **Thyroid Cancer Overview:**
    
    Thyroid cancer is a type of cancer that starts in the thyroid gland. The thyroid is a butterfly-shaped gland located at the base of the neck, just below the Adam's apple. It produces hormones that regulate heart rate, blood pressure, body temperature, and weight.
    
    **Risk Factors for Recurrence:**
    - Age at diagnosis
    - Gender
    - Tumor size and stage
    - Cancer type (papillary, follicular, etc.)
    - Presence of lymph node metastasis
    - Completeness of surgical resection
    - Response to initial treatment
    
    **Follow-up Care:**
    Regular follow-up appointments are crucial for thyroid cancer survivors to monitor for recurrence. These typically include physical exams, blood tests, and imaging studies.
    """)