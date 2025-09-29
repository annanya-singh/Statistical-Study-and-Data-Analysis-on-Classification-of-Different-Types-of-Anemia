#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitive Anemia Classifier
==========================

This script creates a more sensitive anemia classification model that doesn't bias toward one class.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Add the project directory to path
import sys
sys.path.append('/home/preeti/Desktop/projects/anemia_pro')

from utils import create_ratios, cap_outliers, log_transform_skewed

def load_and_preprocess_data():
    """Load and preprocess the CBC data"""
    # Load data
    cbc_data = pd.read_csv('/home/preeti/Desktop/projects/anemia_pro/cbc_data.csv')
    cbc_data['Diagnosis'] = cbc_data['Diagnosis'].astype('category')
    
    # Separate features and target
    X = cbc_data.drop('Diagnosis', axis=1)
    y = cbc_data['Diagnosis']
    
    return X, y

def advanced_preprocessing(X):
    """Apply advanced preprocessing"""
    # Apply feature engineering
    X_processed = create_ratios(X)
    
    # Apply outlier capping
    X_processed = cap_outliers(X_processed)
    
    # Apply log transformation for skewed features
    X_processed = log_transform_skewed(X_processed)
    
    return X_processed

def train_sensitive_model():
    """Train a more sensitive model"""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    X_processed = advanced_preprocessing(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution in training set:")
    print(y_train.value_counts())
    
    # Create a pipeline with SMOTE and Random Forest
    # Using more balanced parameters to avoid overfitting to one class
    sensitive_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # Reduced k_neighbors
        ('classifier', RandomForestClassifier(
            n_estimators=200,           # Moderate number of trees
            max_depth=10,               # Limit depth to prevent overfitting
            min_samples_split=10,       # Increase min samples to split
            min_samples_leaf=5,         # Increase min samples per leaf
            class_weight='balanced',    # Balance class weights
            random_state=42
        ))
    ])
    
    # Train the model
    print("Training sensitive model...")
    sensitive_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = sensitive_pipeline.predict(X_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_dir = '/home/preeti/Desktop/projects/anemia_pro/output_sensitive'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Extract the trained classifier and feature names
    trained_classifier = sensitive_pipeline.named_steps['classifier']
    feature_names = X_processed.columns.tolist()
    
    # Save the model components
    joblib.dump(trained_classifier, os.path.join(model_dir, 'sensitive_rf_classifier.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'sensitive_feature_names.pkl'))
    joblib.dump(sensitive_pipeline, os.path.join(model_dir, 'sensitive_full_pipeline.pkl'))
    
    print(f"\nModel saved to {model_dir}")
    
    return sensitive_pipeline, feature_names

def test_sensitive_model():
    """Test the sensitive model with sample data"""
    # Load the saved model
    model_dir = '/home/preeti/Desktop/projects/anemia_pro/output_sensitive'
    
    try:
        pipeline = joblib.load(os.path.join(model_dir, 'sensitive_full_pipeline.pkl'))
        feature_names = joblib.load(os.path.join(model_dir, 'sensitive_feature_names.pkl'))
        print("Sensitive model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create diverse test samples
    test_samples = [
        # Sample 1 - Should be healthy or normocytic
        {
            'WBC': 7.0, 'LYMp': 35.0, 'NEUTp': 55.0, 'LYMn': 2.5, 'NEUTn': 3.9,
            'RBC': 4.8, 'HGB': 15.0, 'HCT': 42.0, 'MCV': 88.0, 'MCH': 31.0, 'MCHC': 34.0,
            'PLT': 250, 'PDW': 12.0, 'PCT': 0.25
        },
        # Sample 2 - Should be iron deficiency anemia (low HGB, low MCV)
        {
            'WBC': 6.5, 'LYMp': 32.0, 'NEUTp': 58.0, 'LYMn': 2.1, 'NEUTn': 3.8,
            'RBC': 3.8, 'HGB': 9.0, 'HCT': 30.0, 'MCV': 75.0, 'MCH': 24.0, 'MCHC': 30.0,
            'PLT': 300, 'PDW': 14.0, 'PCT': 0.28
        },
        # Sample 3 - Should be macrocytic anemia (high MCV)
        {
            'WBC': 5.8, 'LYMp': 38.0, 'NEUTp': 52.0, 'LYMn': 2.2, 'NEUTn': 3.0,
            'RBC': 3.2, 'HGB': 10.0, 'HCT': 35.0, 'MCV': 110.0, 'MCH': 32.0, 'MCHC': 32.0,
            'PLT': 200, 'PDW': 15.0, 'PCT': 0.22
        }
    ]
    
    for i, sample_data in enumerate(test_samples, 1):
        sample_df = pd.DataFrame([sample_data])
        sample_processed = advanced_preprocessing(sample_df)
        
        # Make prediction
        try:
            prediction = pipeline.predict(sample_processed)[0]
            probabilities = pipeline.predict_proba(sample_processed)[0]
            classes = pipeline.named_steps['classifier'].classes_
            
            print(f"\nTest Sample {i}:")
            print(f"Predicted: {prediction}")
            print("Probabilities:")
            for cls, prob in zip(classes, probabilities):
                print(f"  {cls}: {prob:.4f}")
        except Exception as e:
            print(f"Error predicting sample {i}: {e}")

if __name__ == "__main__":
    # Train the sensitive model
    model, features = train_sensitive_model()
    
    # Test the model
    test_sensitive_model()