#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anemia Prediction Interface
============================

This script provides a simple interface for predicting anemia types based on CBC reports.
It loads the pre-trained Random Forest model and makes predictions on new patient data.

Usage:
    python predict_anemia.py

Or for batch predictions:
    python predict_anemia.py --batch <csv_file_with_cbc_data>
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

# Import the utility function needed for loading the model
from utils import array_to_dataframe, FEATURE_COLUMNS

def get_user_input():
    """Get CBC values from user input"""
    print("\n" + "="*50)
    print("ANEMIA TYPE PREDICTION - CBC INPUT")
    print("="*50)
    
    cbc_data = {}
    for feature in FEATURE_COLUMNS:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                cbc_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    
    return pd.DataFrame([cbc_data])

def predict_single_case(model, cbc_data):
    """Make prediction for a single CBC report"""
    # Make prediction
    prediction = model.predict(cbc_data)[0]
    probabilities = model.predict_proba(cbc_data)[0]
    
    # Get class names
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        # For pipeline, we need to access the final estimator
        classes = model.named_steps['model'].classes_
    
    # Create probability dictionary
    prob_dict = dict(zip(classes, probabilities))
    
    return prediction, prob_dict

def display_results(prediction, probabilities):
    """Display prediction results"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Anemia Type: {prediction}")
    print("\nPrediction Probabilities:")
    print("-" * 30)
    
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for anemia_type, prob in sorted_probs:
        print(f"{anemia_type:<30}: {prob:.4f} ({prob*100:.2f}%)")

def load_batch_data(file_path):
    """Load CBC data from a CSV file for batch prediction"""
    try:
        data = pd.read_csv(file_path)
        # Check if all required columns are present
        missing_cols = set(FEATURE_COLUMNS) - set(data.columns)
        if missing_cols:
            print(f"Error: Missing columns in the input file: {missing_cols}")
            return None
        return data[FEATURE_COLUMNS]
    except Exception as e:
        print(f"Error loading batch data: {e}")
        return None

def save_batch_predictions(predictions, probabilities, input_data, output_file):
    """Save batch predictions to a CSV file"""
    # Create results dataframe
    results = input_data.copy()
    results['Predicted_Anemia_Type'] = predictions
    
    # Add probability columns for each class
    classes = list(probabilities[0].keys()) if probabilities else []
    for cls in classes:
        results[f'Probability_{cls.replace(" ", "_")}'] = [prob[cls] for prob in probabilities]
    
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
    
    # Check if model is available or needs to be trained
    if not MODEL_AVAILABLE:
        print("Error: Model not found. Please run 'run_analysis.py' first to train the model.")
        return
    
    # Fit the model if not already fitted
    try:
        # Check if model is already fitted by trying to access classes_
        _ = rf_full_pipeline.classes_
    except AttributeError:
        # Model not fitted, fit it now
        print("Fitting the model...")
        rf_full_pipeline.fit(X_train, y_train)
    
    model = rf_full_pipeline
    
    if args.batch:
        # Batch prediction mode
        print(f"Performing batch prediction on: {args.batch}")
        cbc_data = load_batch_data(args.batch)
        
        if cbc_data is not None:
            predictions = []
            probabilities = []
            
            for idx, row in cbc_data.iterrows():
                pred, probs = predict_single_case(model, pd.DataFrame([row]))
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
                                 pd.read_csv(args.batch), args.output)
        else:
            print("Failed to load batch data.")
    else:
        # Interactive mode
        while True:
            # Get user input
            cbc_data = get_user_input()
            
            # Make prediction
            prediction, probabilities = predict_single_case(model, cbc_data)
            
            # Display results
            display_results(prediction, probabilities)
            
            # Ask if user wants to continue
            cont = input("\nWould you like to predict another case? (y/n): ").lower()
            if cont != 'y' and cont != 'yes':
                break
        
        print("\nThank you for using the Anemia Prediction System!")

if __name__ == "__main__":
    main()