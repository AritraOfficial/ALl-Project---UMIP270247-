import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('vehicle_price_model.h5')
scaler = joblib.load('scaler.pkl')

# Set page config and styling
st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #008CBA; color: white; font-weight: bold; }
    .stTabs [role="tab"] { font-size: 18px; }
    .highlight { font-size: 26px; font-weight: bold; color: green; padding: 10px 0; }
    .inr {
        font-size: 20px;
        font-weight: bold;
        color: #444;
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ffeeba;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("image.png", width=250)
    st.title("📊 Vehicle Options")
    st.markdown("🔧 Customize your vehicle and predict its price in **USD & INR**.")
    st.markdown("---")
    st.markdown("Made by **Aritra**  \n📬 aritra.work.official@gmail.com")

# Tabs
tab1, tab2 = st.tabs(["🚗 Predict Price", "📘 Info & Help"])

# Feature order from training
expected_columns = ['make', 'model', 'year', 'mileage', 'fuel', 'transmission', 'trim',
                    'body', 'cylinders', 'doors', 'exterior_color', 'interior_color',
                    'drivetrain', 'engine']

# Dropdown options
make_options = ['Ford', 'Hyundai', 'Kia', 'Jeep', 'Toyota', 'RAM', 'Chevrolet', 'Nissan', 'BMW', 'Honda']
model_options = ['Wagoneer', 'Grand Cherokee', 'Tucson', 'Compass', 'F-150', 'Silverado 1500', 'Mustang', 'Rogue']
trim_options = ['Base', 'Limited', 'Sport', 'Touring', 'XLE', 'SE', 'GT']
engine_options = sorted([
    '24V GDI DOHC Twin Turbo', '16V MPFI DOHC', 'DOHC', 'EcoBoost', 'VVT Turbo Diesel',
    '16V GDI DOHC Turbo Hybrid', '24V MPFI DOHC Hybrid', '6.2L V-8 gasoline direct injection',
    'Turbo Premium Unleaded I-4', '5.3L V-8', 'Electric', 'High Output 6.7L I-6 diesel',
    'Hybrid CVVT', '3.6L V-6 DOHC', '16V PDI DOHC Turbo', '2.5L I-4 DOHC CVVT'
])

with tab1:
    st.title("🚗 Vehicle Price Predictor")
    st.markdown("Fill in the vehicle details to estimate the price in **USD** and **INR**.")

    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Make", make_options)
        model_input = st.selectbox("Model", model_options)
        year = st.slider("Year", 1990, 2025, 2024)
        mileage = st.number_input("Mileage (in miles)", min_value=0)
        fuel = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Electric', 'Hybrid'])
        transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'CVT'])
        trim = st.selectbox("Trim Level", trim_options)

    with col2:
        body = st.selectbox("Body Type", ['SUV', 'Sedan', 'Hatchback', 'Truck', 'Van'])
        cylinders = st.slider("Cylinders", 2, 16, 4)
        doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
        exterior_color = st.text_input("Exterior Color", "White")
        interior_color = st.text_input("Interior Color", "Black")
        drivetrain = st.selectbox("Drivetrain", ['FWD', 'RWD', 'AWD', '4WD'])
        engine = st.selectbox("Engine Type", engine_options)

    if st.button("🔍 Predict Price"):
        input_data = pd.DataFrame([{
            'make': make, 'model': model_input, 'year': year, 'mileage': mileage,
            'fuel': fuel, 'transmission': transmission, 'trim': trim, 'body': body,
            'cylinders': cylinders, 'doors': doors, 'exterior_color': exterior_color,
            'interior_color': interior_color, 'drivetrain': drivetrain, 'engine': engine
        }])[expected_columns]

        for col in input_data.columns:
            input_data[col] = input_data[col].astype(str).astype('category').cat.codes

        input_array = scaler.transform(input_data.values)
        predicted_price = model.predict(input_array)[0][0]

        # Clamp prediction to realistic range based on dataset
        predicted_price = max(15000, min(predicted_price, 120000))

        # Convert to INR
        usd = round(predicted_price, 2)
        inr_exact = int(usd * 83)
        min_inr = int(inr_exact * 0.95)
        max_inr = int(inr_exact * 1.05)

        st.markdown(f"<div class='highlight'>💰 Estimated Price: ${usd:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='inr'>🇮🇳 Estimated INR Range: ₹{min_inr:,} – ₹{max_inr:,}</div>", unsafe_allow_html=True)
        st.markdown(
    "<small><i>Note: This prediction is based on the model's training data and may vary from actual market prices. "
    "Factors like vehicle condition, local market, and specific features can affect the actual price.</i></small>",
    unsafe_allow_html=True
)

with tab2:
    st.title("📘 Vehicle Info & Help")
    st.markdown("""
    ### 🔧 How to Use:
    - Fill all fields under "Predict Price" tab.
    - Click on **Predict Price** to get the estimate.
    - The model gives price in both **USD** and approximate **INR** range.

    ### 🧾 Field Guide:
    - **Make & Model**: Brand and model name.
    - **Trim**: Variant with extra features.
    - **Engine**: Engine configuration/type (e.g., EcoBoost, V6).
    - **Fuel**: Fuel type like Gasoline or Hybrid.
    - **Transmission**: Auto, Manual, or CVT.
    - **Drivetrain**: Power delivery - FWD, RWD, AWD.
    - **Mileage**: Vehicle distance used (in miles).

    ### 💡 Notes:
    - INR value assumes ₹83/USD with ±5% range.
    - This tool is based on machine learning predictions and may vary from actual dealer prices.

    ### 💌 Contact:
    Email: aritra.work.official@gmail.com
    """)
