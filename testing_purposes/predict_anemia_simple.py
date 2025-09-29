#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anemia Prediction Interface
============================

This script provides a simple interface for predicting anemia types based on CBC reports.
It loads the pre-trained Random Forest model and makes predictions on new patient data.

Usage:
    python predict_anemia_simple.py

Or for batch predictions:
    python predict_anemia_simple.py --batch <csv_file_with_cbc_data>
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import array_to_dataframe, create_ratios, cap_outliers, log_transform_skewed

# Define the expected feature columns
ORIGINAL_FEATURE_COLUMNS = [
    'WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 
    'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT'
]

def preprocess_cbc_data(cbc_data):
    """Apply the same preprocessing steps used during training"""
    # Apply the same transformations as in the training pipeline
    cbc_data = array_to_dataframe(cbc_data)
    cbc_data = create_ratios(cbc_data)
    cbc_data = cap_outliers(cbc_data)
    cbc_data = log_transform_skewed(cbc_data)
    return cbc_data

def get_user_input():
    """Get CBC values from user input"""
    print("\n" + "="*50)
    print("ANEMIA TYPE PREDICTION - CBC INPUT")
    print("="*50)
    
    cbc_data = {}
    for feature in ORIGINAL_FEATURE_COLUMNS:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                cbc_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return pd.DataFrame([cbc_data])

def predict_single_case(model, feature_names, cbc_data):
    """Make prediction for a single CBC report"""
    # Preprocess the data
    processed_data = preprocess_cbc_data(cbc_data[ORIGINAL_FEATURE_COLUMNS])
    
    # Select only the features the model was trained on
    processed_data = processed_data[feature_names]
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    probabilities = model.predict_proba(processed_data)[0]
    
    # Get class names (assuming they're stored in the model or we load them separately)
    # For this simplified version, we'll assume they're available
    return prediction, probabilities

def display_results(prediction, probabilities, class_names):
    """Display prediction results"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Anemia Type: {prediction}")
    print("\nPrediction Probabilities:")
    print("-" * 30)
    
    # Sort by probability
    sorted_probs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
    for anemia_type, prob in sorted_probs:
        print(f"{anemia_type:<30}: {prob:.4f} ({prob*100:.2f}%)")

def load_batch_data(file_path):
    """Load CBC data from a CSV file for batch prediction"""
    try:
        data = pd.read_csv(file_path)
        # Check if all required columns are present
        missing_cols = set(ORIGINAL_FEATURE_COLUMNS) - set(data.columns)
        if missing_cols:
            print(f"Error: Missing columns in the input file: {missing_cols}")
            return None
        return data
    except Exception as e:
        print(f"Error loading batch data: {e}")
        return None

def save_batch_predictions(predictions, probabilities, input_data, class_names, output_file):
    """Save batch predictions to a CSV file"""
    # Create results dataframe
    results = input_data.copy()
    results['Predicted_Anemia_Type'] = predictions
    
    # Add probability columns for each class
    for i, cls in enumerate(class_names):
        results[f'Probability_{cls.replace(" ", "_")}'] = [prob[i] for prob in probabilities]
    
    try:
        results.to_csv(output_file, index=False)
        print(f"Batch predictions saved to: {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Anemia Type Prediction from CBC Data')
    parser.add_argument('--batch', type=str, help='CSV file with CBC data for batch prediction')
    parser.add_argument('--output', type=str, default='anemia_predictions.csv', 
                        help='Output file for batch predictions (default: anemia_predictions.csv)')
    
    args = parser.parse_args()
    
    # Load the trained model and feature names
    try:
        model = joblib.load('output_plots_advanced/rf_classifier.pkl')
        feature_names = joblib.load('output_plots_advanced/processed_feature_names.pkl')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'run_analysis.py' first to train and save the model.")
        return
    
    # For this example, we'll use the sample data we saved to get class names
    try:
        sample_data = joblib.load('output_plots_advanced/sample_preprocessed_data.pkl')
        # We'll need to get class names from the original training data or model
        # In a real implementation, these would be saved with the model
        class_names = ['Healthy', 'Iron deficiency anemia', 'Leukemia', 
                      'Leukemia with thrombocytopenia', 'Macrocytic anemia',
                      'Normocytic hypochromic anemia', 'Normocytic normochromic anemia',
                      'Other microcytic anemia', 'Thrombocytopenia']
    except:
        # Fallback class names
        class_names = ['Healthy', 'Iron deficiency anemia', 'Leukemia', 
                      'Leukemia with thrombocytopenia', 'Macrocytic anemia',
                      'Normocytic hypochromic anemia', 'Normocytic normochromic anemia',
                      'Other microcytic anemia', 'Thrombocytopenia']
    
    if args.batch:
        # Batch prediction mode
        print(f"Performing batch prediction on: {args.batch}")
        cbc_data = load_batch_data(args.batch)
        
        if cbc_data is not None:
            predictions = []
            probabilities = []
            
            for idx, row in cbc_data.iterrows():
                pred, probs = predict_single_case(model, feature_names, pd.DataFrame([row]))
                predictions.append(pred)
                probabilities.append(probs)
            
            # Display summary
            print(f"\nProcessed {len(predictions)} cases.")
            unique, counts = np.unique(predictions, return_counts=True)
            print("\nPrediction Summary:")
            for anemia_type, count in zip(unique, counts):
                print(f"  {anemia_type}: {count}")
            
            # Save results
            save_batch_predictions(predictions, probabilities, 
                                 cbc_data, class_names, args.output)
        else:
            print("Failed to load batch data.")
    else:
        # Interactive mode
        while True:
            # Get user input
            cbc_data = get_user_input()
            
            # Make prediction
            try:
                prediction, probabilities = predict_single_case(model, feature_names, cbc_data)
                
                # Display results
                display_results(prediction, probabilities, class_names)
            except Exception as e:
                print(f"Error making prediction: {e}")
                print("Please check your input values and try again.")
            
            # Ask if user wants to continue
            cont = input("\nWould you like to predict another case? (y/n): ").lower()
            if cont != 'y' and cont != 'yes':
                break
        
        print("\nThank you for using the Anemia Prediction System!")

if __name__ == "__main__":
    main()