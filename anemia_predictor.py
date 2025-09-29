#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI for Anemia Classification
=====================================

This Streamlit app provides a user-friendly interface for predicting anemia types
based on CBC reports using the trained machine learning model.

Usage:
    streamlit run anemia_predictor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_ratios, cap_outliers, log_transform_skewed
import os

# Configuration
MODEL_PATH = '/home/preeti/Desktop/projects/anemia_pro/output_sensitive/sensitive_full_pipeline.pkl'
FEATURE_NAMES_PATH = '/home/preeti/Desktop/projects/anemia_pro/output_sensitive/sensitive_feature_names.pkl'

# Set page configuration
st.set_page_config(
    page_title="Anemia Classifier",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stAlert {
        background-color: #e1f5fe;
        border: 1px solid #b3e5fc;
    }
    .prediction-box {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .high-probability {
        background-color: #c8e6c9;
        font-weight: bold;
    }
    .medium-probability {
        background-color: #fff9c4;
    }
    .low-probability {
        background-color: #ffcdd2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and feature names"""
    try:
        pipeline = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        return pipeline, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_input_data(input_data):
    """Apply the same preprocessing steps used during training"""
    # Apply feature engineering
    processed_data = create_ratios(input_data)
    
    # Apply outlier capping
    processed_data = cap_outliers(processed_data)
    
    # Apply log transformation for skewed features
    processed_data = log_transform_skewed(processed_data)
    
    return processed_data

def create_visualizations(input_data, prediction, probabilities, class_names):
    """Create visualizations for the input data"""
    st.subheader("📊 Data Visualization")
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for CBC values
        st.write("**CBC Values Overview**")
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        
        # Normalize values for radar chart (0-1 scale)
        normalized_data = (input_data.iloc[0] - input_data.iloc[0].min()) / (input_data.iloc[0].max() - input_data.iloc[0].min())
        normalized_data = normalized_data.fillna(0)
        
        # Only use original features for radar chart
        original_features = [col for col in input_data.columns if col in 
                           ['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 
                            'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']]
        
        if len(original_features) > 0:
            values = normalized_data[original_features].values
            angles = np.linspace(0, 2 * np.pi, len(original_features), endpoint=False).tolist()
            values = np.concatenate((values, [values[0]]))  # Close the circle
            angles += angles[:1]  # Close the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(original_features, fontsize=8)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
        else:
            st.write("Unable to create radar chart with available features.")
    
    with col2:
        # Probability distribution
        st.write("**Prediction Probabilities**")
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = np.arange(len(class_names))
        colors = plt.cm.viridis(probabilities)
        bars = ax.barh(y_pos, probabilities, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities')
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.3f}', ha='left', va='center')
        
        st.pyplot(fig)

def main():
    """Main Streamlit app"""
    st.title("🩸 Anemia Type Classifier")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This application predicts anemia types based on Complete Blood Count (CBC) reports
    using a machine learning model trained on real medical data.
    
    **Features used:**
    - WBC, LYMp, NEUTp, LYMn, NEUTn
    - RBC, HGB, HCT, MCV, MCH, MCHC
    - PLT, PDW, PCT
    - Engineered features: NLR
    """)
    
    st.sidebar.title("Instructions")
    st.sidebar.info("""
    1. Enter CBC values in the input form
    2. Click 'Predict Anemia Type'
    3. View prediction results and visualizations
    """)
    
    # Load model
    pipeline, feature_names = load_model()
    
    if pipeline is None or feature_names is None:
        st.error("Failed to load the model. Please check the model files.")
        return
    
    # Main content
    st.header("Enter CBC Report Data")
    
    # Create input form
    with st.form("cbc_form"):
        st.subheader("Blood Count Values")
        
        # Create columns for better organization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wbc = st.number_input("WBC (10^9/L)", min_value=0.0, max_value=100.0, value=7.0, step=0.1)
            rbc = st.number_input("RBC (10^12/L)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
            hgb = st.number_input("HGB (g/dL)", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
            hct = st.number_input("HCT (%)", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
            mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=200.0, value=90.0, step=0.1)
            
        with col2:
            lymp = st.number_input("LYMp (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
            lympn = st.number_input("LYMn (10^9/L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
            mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)
            neutp = st.number_input("NEUTp (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            
        with col3:
            neutn = st.number_input("NEUTn (10^9/L)", min_value=0.0, max_value=20.0, value=4.0, step=0.1)
            plt_val = st.number_input("PLT (10^9/L)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
            pdw = st.number_input("PDW (fL)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
            pct = st.number_input("PCT (%)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Anemia Type")
    
    # Prediction section
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([{
            'WBC': wbc, 'LYMp': lymp, 'NEUTp': neutp, 'LYMn': lympn, 'NEUTn': neutn,
            'RBC': rbc, 'HGB': hgb, 'HCT': hct, 'MCV': mcv, 'MCH': mch, 'MCHC': mchc,
            'PLT': plt_val, 'PDW': pdw, 'PCT': pct
        }])
        
        # Preprocess the input data
        processed_data = preprocess_input_data(input_data)
        
        # Make prediction
        try:
            prediction = pipeline.predict(processed_data)[0]
            probabilities = pipeline.predict_proba(processed_data)[0]
            class_names = pipeline.named_steps['classifier'].classes_
            
            # Display results
            st.header("📋 Prediction Results")
            
            # Main prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Anemia Type: <span style="color: #1f77b4;">{prediction}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilities table
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Anemia Type': class_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Style the dataframe based on probability values
            def style_probabilities(val):
                if val > 0.7:
                    return 'background-color: #c8e6c9'  # High probability (green)
                elif val > 0.3:
                    return 'background-color: #fff9c4'  # Medium probability (yellow)
                else:
                    return 'background-color: #ffcdd2'   # Low probability (red)
            
            styled_df = prob_df.style.applymap(style_probabilities, subset=['Probability'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Create visualizations
            create_visualizations(input_data, prediction, probabilities, class_names)
            
            # Additional information
            st.header("ℹ️ Additional Information")
            
            # Risk assessment
            max_prob = max(probabilities)
            if max_prob > 0.8:
                st.success("✅ High confidence prediction")
            elif max_prob > 0.6:
                st.warning("⚠️ Moderate confidence prediction")
            else:
                st.error("❌ Low confidence prediction - please consult a healthcare professional")
            
            # Show input data summary
            st.subheader("Input Data Summary")
            st.dataframe(input_data.T.style.format("{:.2f}"), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()