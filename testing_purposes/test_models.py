#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Testing Script
====================

This script tests the trained models to understand prediction behavior.
"""

import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('/home/preeti/Desktop/projects/anemia_pro')

from utils import create_ratios, cap_outliers, log_transform_skewed

def test_model_predictions():
    """Test model predictions with sample data"""
    
    # Load models
    try:
        rf_model = joblib.load('/home/preeti/Desktop/projects/anemia_pro/output_plots_advanced/rf_model.pkl')
        rf_classifier = joblib.load('/home/preeti/Desktop/projects/anemia_pro/output_plots_advanced/rf_classifier.pkl')
        processed_feature_names = joblib.load('/home/preeti/Desktop/projects/anemia_pro/output_plots_advanced/processed_feature_names.pkl')
        print("Models loaded successfully")
        print(f"RF Model classes: {rf_model.classes_}")
        print(f"RF Classifier classes: {rf_classifier.classes_}")
        print(f"Processed feature names: {processed_feature_names}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Create sample data that should predict different classes
    sample_data_1 = pd.DataFrame([{
        'WBC': 8.5, 'LYMp': 30.5, 'NEUTp': 55.2, 'LYMn': 2.6, 'NEUTn': 4.7,
        'RBC': 4.2, 'HGB': 12.5, 'HCT': 36.2, 'MCV': 86.2, 'MCH': 29.8, 'MCHC': 34.6,
        'PLT': 210, 'PDW': 13.2, 'PCT': 0.22
    }])
    
    sample_data_2 = pd.DataFrame([{
        'WBC': 5.2, 'LYMp': 35.6, 'NEUTp': 56.9, 'LYMn': 2.0, 'NEUTn': 3.2,
        'RBC': 4.31, 'HGB': 11.0, 'HCT': 36.3, 'MCV': 84.3, 'MCH': 25.5, 'MCHC': 30.3,
        'PLT': 171, 'PDW': 13.1, 'PCT': 0.15
    }])
    
    print("\nSample Data 1:")
    print(sample_data_1)
    
    print("\nSample Data 2:")
    print(sample_data_2)
    
    # Preprocess the data
    processed_data_1 = preprocess_input_data(sample_data_1)
    processed_data_2 = preprocess_input_data(sample_data_2)
    
    print("\nProcessed Data 1:")
    print(processed_data_1.columns.tolist())
    
    print("\nProcessed Data 2:")
    print(processed_data_2.columns.tolist())
    
    # Make predictions with RF model
    try:
        # For rf_model, we need to use the full pipeline
        pred_1_rf = rf_model.predict(sample_data_1)
        prob_1_rf = rf_model.predict_proba(sample_data_1)
        
        pred_2_rf = rf_model.predict(sample_data_2)
        prob_2_rf = rf_model.predict_proba(sample_data_2)
        
        print(f"\nRF Model Predictions:")
        print(f"Sample 1: {pred_1_rf[0]}")
        print(f"Sample 1 probabilities: {dict(zip(rf_model.classes_, prob_1_rf[0]))}")
        print(f"Sample 2: {pred_2_rf[0]}")
        print(f"Sample 2 probabilities: {dict(zip(rf_model.classes_, prob_2_rf[0]))}")
    except Exception as e:
        print(f"Error with RF model: {e}")
    
    # Make predictions with RF classifier
    try:
        # For rf_classifier, we need to select the right features
        model_input_1 = processed_data_1[processed_feature_names]
        model_input_2 = processed_data_2[processed_feature_names]
        
        pred_1_classifier = rf_classifier.predict(model_input_1)
        prob_1_classifier = rf_classifier.predict_proba(model_input_1)
        
        pred_2_classifier = rf_classifier.predict(model_input_2)
        prob_2_classifier = rf_classifier.predict_proba(model_input_2)
        
        print(f"\nRF Classifier Predictions:")
        print(f"Sample 1: {pred_1_classifier[0]}")
        print(f"Sample 1 probabilities: {dict(zip(rf_classifier.classes_, prob_1_classifier[0]))}")
        print(f"Sample 2: {pred_2_classifier[0]}")
        print(f"Sample 2 probabilities: {dict(zip(rf_classifier.classes_, prob_2_classifier[0]))}")
    except Exception as e:
        print(f"Error with RF classifier: {e}")

def preprocess_input_data(input_data):
    """Apply the same preprocessing steps used during training"""
    # Apply feature engineering
    processed_data = create_ratios(input_data)
    
    # Apply outlier capping
    processed_data = cap_outliers(processed_data)
    
    # Apply log transformation for skewed features
    processed_data = log_transform_skewed(processed_data)
    
    return processed_data

if __name__ == "__main__":
    test_model_predictions()