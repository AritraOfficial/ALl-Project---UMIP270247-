import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Mobile Phone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .feature-section {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-section {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-header'>Mobile Phone Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
This app predicts the price range of a mobile phone based on its specifications. 
Fill in the details below to get a prediction of where your phone would be priced in the market.
""")

@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('mobile_price_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        price_range_dict = pickle.load(open('price_range_dict.pkl', 'rb'))
        return model, scaler, feature_names, price_range_dict
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model is trained and saved correctly.")
        return None, None, None, None

model, scaler, feature_names, price_range_dict = load_model()

# Sidebar for instructions
with st.sidebar:
    st.title("📘 Instructions")
    st.info("""
    1. Fill in the specifications of the mobile phone
    2. Click on 'Predict Price Range' to get the result
    3. The prediction will show the estimated price range and confidence
    """)
    
    st.title("🔍 About")
    st.write("""
    This app uses a machine learning model trained on mobile phone specifications to predict price ranges:
    - Low: ₹5,000-12,000
    - Medium: ₹12,000-20,000
    - High: ₹20,000-35,000
    - Very High: ₹35,000+
    """)

tab1, tab2 = st.tabs(["Prediction", "Feature Importance"])

with tab1:
    st.markdown("<h2 class='sub-header'>Enter Phone Specifications</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with st.container():
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Core Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ram = st.slider("RAM (MB)", min_value=256, max_value=4000, value=2000, step=256, 
                            help="Random Access Memory in Megabytes")
            int_memory = st.slider("Internal Memory (GB)", min_value=4, max_value=128, value=32, step=4,
                                help="Internal storage capacity in Gigabytes")
            battery_power = st.slider("Battery Capacity (mAh)", min_value=500, max_value=2500, value=1500, step=100,
                                    help="Battery capacity in milliampere-hours")
            
        with col2:
            n_cores = st.slider("Processor Cores", min_value=1, max_value=8, value=4, step=1,
                                help="Number of processor cores")
            clock_speed = st.slider("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.8, step=0.1,
                                    help="Processor clock speed in GHz")
            mobile_wt = st.slider("Weight (grams)", min_value=80, max_value=250, value=160, step=5,
                                help="Mobile phone weight in grams")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Display")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sc_h = st.slider("Screen Height (cm)", min_value=5, max_value=20, value=12, step=1,
                            help="Screen height in centimeters")
            px_height = st.slider("Pixel Height", min_value=480, max_value=2560, value=1280, step=80,
                                help="Screen resolution height in pixels")
            touch_screen = st.selectbox("Touch Screen", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                    help="Does the phone have a touch screen")
            
        with col2:
            sc_w = st.slider("Screen Width (cm)", min_value=3, max_value=15, value=6, step=1,
                            help="Screen width in centimeters")
            px_width = st.slider("Pixel Width", min_value=320, max_value=1600, value=720, step=80,
                                help="Screen resolution width in pixels")
            m_dep = st.slider("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                            help="Mobile phone thickness in centimeters")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Camera")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pc = st.slider("Primary Camera (MP)", min_value=0, max_value=20, value=12, step=1,
                        help="Primary camera resolution in megapixels")
            
        with col2:
            fc = st.slider("Front Camera (MP)", min_value=0, max_value=20, value=5, step=1,
                        help="Front camera resolution in megapixels")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Connectivity & Other Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bluetooth = st.selectbox("Bluetooth", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                    help="Bluetooth connectivity")
            
        with col2:
            wifi = st.selectbox("WiFi", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                help="WiFi connectivity")
            
        with col3:
            three_g = st.selectbox("3G", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                help="3G connectivity")
            
        with col4:
            four_g = st.selectbox("4G", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                help="4G connectivity")
            
        col1, col2 = st.columns(2)
        
        with col1:
            dual_sim = st.selectbox("Dual SIM", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                                help="Dual SIM support")
            
        with col2:
            talk_time = st.slider("Talk Time (hours)", min_value=2, max_value=24, value=12, step=1,
                                help="Battery talk time in hours")
        st.markdown("</div>", unsafe_allow_html=True)
       
        predict_button = st.button("Predict Price Range", type="primary", use_container_width=True)
    
        screen_area = sc_h * sc_w
        resolution = px_height * px_width
        
        if predict_button:
           
            input_data = {
                'battery_power': battery_power,
                'blue': bluetooth,
                'clock_speed': clock_speed,
                'dual_sim': dual_sim,
                'fc': fc,
                'four_g': four_g,
                'int_memory': int_memory,
                'm_dep': m_dep,
                'mobile_wt': mobile_wt,
                'n_cores': n_cores,
                'pc': pc,
                'px_height': px_height,
                'px_width': px_width,
                'ram': ram,
                'sc_h': sc_h,
                'sc_w': sc_w,
                'talk_time': talk_time,
                'three_g': three_g,
                'touch_screen': touch_screen,
                'wifi': wifi,
                'screen_area': screen_area,
                'resolution': resolution
            }
            
            try:
                
                input_df = pd.DataFrame([input_data], columns=feature_names)
                
               
                input_scaled = scaler.transform(input_df)
                
               
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                confidence = probabilities[prediction] * 100
                
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                st.subheader("Prediction Result")
                
                price_emojis = {0: "🟢", 1: "🔵", 2: "🟠", 3: "🔴"}
                
                st.markdown(f"""
                <h2 style='font-size: 2rem;'>{price_emojis[prediction]} Price Range: <span style='color:#2E7D32;'>{price_range_dict[prediction]}</span></h2>
                <p style='font-size: 1.2rem;'>Confidence: <span style='font-weight:bold;'>{confidence:.1f}%</span></p>
                """, unsafe_allow_html=True)
                
               
                fig, ax = plt.subplots(figsize=(10, 5))
                ranges = [price_range_dict[i] for i in range(4)]
                colors = ['#2E7D32', '#1976D2', '#FF9800', '#D32F2F']
                
                bars = ax.bar(ranges, probabilities * 100, color=[colors[i] for i in range(4)])
                
               
                bars[prediction].set_color('darkgreen')
                
                ax.set_ylabel('Probability (%)')
                ax.set_title('Prediction Probabilities for Each Price Range')
                ax.set_ylim(0, 100)
                
             
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{probabilities[i]*100:.1f}%',
                            ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
    st.write("""
    This visualization shows which features are most important for determining the price range of a mobile phone,
    according to our machine learning model.
    """)
    
    if model is not None:
       
        if hasattr(model, 'feature_importances_'):
          
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)
            
          
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = sns.color_palette("viridis", len(feature_importance))
            
            bars = ax.barh(feature_importance['Feature'][:15], 
                          feature_importance['Importance'][:15],
                          color=colors[:15])
            
            ax.set_title('Top 15 Feature Importances', fontsize=15)
            ax.set_xlabel('Importance')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            st.subheader("Feature Importance Table")
            st.dataframe(feature_importance.head(15), width=800)
            
            st.markdown("""
            ### Interpretation
            
            The features with higher importance have a stronger influence on the predicted price range. 
            As we can see, RAM, battery power, and screen resolution appear to be the most important factors
            that determine a mobile phone's price range.
            """)
        else:
            st.warning("This model doesn't provide feature importances.")
    else:
        st.error("Model not loaded correctly. Please check if model files exist.")


st.markdown("<div class='footer'>Mobile Price Prediction App | Created with Aritra Mukherjee.</div>", unsafe_allow_html=True)