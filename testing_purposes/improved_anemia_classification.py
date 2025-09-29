#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Anemia Classification with Proper Validation
=====================================================

This script implements a statistically robust anemia classification system with:
1. Proper regularization to prevent overfitting
2. Nested cross-validation for unbiased evaluation
3. Comparison of feature subsets
4. Advanced statistical validation

Usage:
    python improved_anemia_classification.py
"""

import pandas as pd
import numpy as np
import os
import warnings
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Core ML libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, 
    GridSearchCV, RandomizedSearchCV, train_test_split,
    permutation_test_score, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    balanced_accuracy_score, accuracy_score, make_scorer
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.calibration import CalibratedClassifierCV

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
DATA_FILE = '/home/preeti/Desktop/projects/anemia_pro/cbc_data.csv'
OUTPUT_DIR = '/home/preeti/Desktop/projects/anemia_pro/output_improved'

# Setup environment
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style="whitegrid")

class ImprovedAnemiaClassifier:
    """Improved Anemia Classification System with Proper Validation"""
    
    def __init__(self):
        self.best_model = None
        self.feature_names = None
        self.class_names = None
        
    def load_and_prepare_data(self):
        """Load and prepare the CBC data"""
        print("--- [1/8] Loading and analyzing data ---")
        try:
            cbc_data = pd.read_csv(DATA_FILE)
        except FileNotFoundError:
            print(f"FATAL ERROR: The file '{DATA_FILE}' was not found.")
            exit()
            
        print(f"Successfully loaded data. Shape: {cbc_data.shape}")
        
        # Missing Value Analysis
        missing_values = cbc_data.isnull().sum()
        missing_percent = (missing_values / len(cbc_data)) * 100
        missing_df = pd.DataFrame({'count': missing_values, 'percent': missing_percent})
        print("\nMissing Value Report:\n", missing_df[missing_df['count'] > 0])
        
        # Prepare data for modeling
        cbc_data['Diagnosis'] = cbc_data['Diagnosis'].astype('category')
        X = cbc_data.drop('Diagnosis', axis=1)
        y = cbc_data['Diagnosis']
        self.feature_names = X.select_dtypes(include=np.number).columns.tolist()
        self.class_names = y.unique()
        
        return X, y
    
    def advanced_preprocessing(self, X):
        """Advanced preprocessing with feature engineering"""
        print("\n--- [2/8] Advanced preprocessing ---")
        
        # Custom preprocessing functions
        def create_ratios(df):
            """Feature Engineering: Create clinically relevant ratios"""
            df_copy = df.copy()
            # Neutrophil-to-Lymphocyte Ratio (NLR)
            df_copy['NLR'] = df_copy['NEUTn'] / (df_copy['LYMn'] + 1e-6)
            # Platelet-to-Lymphocyte Ratio (PLR)
            df_copy['PLR'] = df_copy['PLT'] / (df_copy['LYMn'] + 1e-6)
            # Mean Corpuscular Hemoglobin Density
            df_copy['MCHD'] = df_copy['MCH'] / df_copy['MCV']
            return df_copy
        
        def cap_outliers(df, iqr_factor=1.5):
            """Outlier Handling: Cap outliers using the IQR method"""
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=np.number).columns:
                q1 = df_copy[col].quantile(0.25)
                q3 = df_copy[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (iqr * iqr_factor)
                upper_bound = q3 + (iqr * iqr_factor)
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
            return df_copy
        
        def log_transform_skewed(df, skew_threshold=0.75):
            """Skewness Correction: Apply log transform to highly skewed features"""
            df_copy = df.copy()
            for col in df_copy.select_dtypes(include=np.number).columns:
                if abs(df_copy[col].skew()) > skew_threshold:
                    # Use log1p to handle zero values gracefully
                    df_copy[col] = np.log1p(df_copy[col])
            return df_copy
        
        # Apply preprocessing steps
        X_processed = create_ratios(X)
        X_processed = cap_outliers(X_processed)
        X_processed = log_transform_skewed(X_processed)
        
        return X_processed
    
    def nested_cross_validation(self, X, y):
        """Perform nested cross-validation to avoid optimistic bias"""
        print("\n--- [3/8] Nested cross-validation ---")
        
        # Outer CV for model evaluation
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Define the model with proper regularization
        rf = RandomForestClassifier(random_state=42)
        
        # Define hyperparameter space with proper constraints
        param_grid = {
            'n_estimators': [300, 500],  # Increased trees
            'max_depth': [10, 12, 15],   # Limited depth to prevent overfitting
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Nested CV with grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=inner_cv, 
            scoring='balanced_accuracy', n_jobs=-1, verbose=0
        )
        
        # Perform nested cross-validation
        nested_scores = cross_val_score(
            grid_search, X, y, cv=outer_cv, 
            scoring='balanced_accuracy', n_jobs=-1
        )
        
        print(f"Nested CV Balanced Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f})")
        
        # Also get accuracy scores
        nested_scores_acc = cross_val_score(
            grid_search, X, y, cv=outer_cv, 
            scoring='accuracy', n_jobs=-1
        )
        
        print(f"Nested CV Accuracy: {nested_scores_acc.mean():.4f} (+/- {nested_scores_acc.std() * 2:.4f})")
        
        return nested_scores, nested_scores_acc
    
    def feature_selection_comparison(self, X, y):
        """Compare different feature subsets"""
        print("\n--- [4/8] Feature selection comparison ---")
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # 1. All features (17 total after engineering)
        selector_all = SelectKBest(score_func=f_classif, k='all')
        X_all = selector_all.fit_transform(X, y)
        selected_features_all = np.array(feature_names)
        
        # 2. Top 15 features
        selector_15 = SelectKBest(score_func=f_classif, k=15)
        X_15 = selector_15.fit_transform(X, y)
        selected_features_15 = np.array(feature_names)[selector_15.get_support()]
        
        # 3. Top 10 features
        selector_10 = SelectKBest(score_func=f_classif, k=10)
        X_10 = selector_10.fit_transform(X, y)
        selected_features_10 = np.array(feature_names)[selector_10.get_support()]
        
        print(f"All features ({len(selected_features_all)}): {selected_features_all}")
        print(f"Top 15 features: {selected_features_15}")
        print(f"Top 10 features: {selected_features_10}")
        
        # Cross-validate each feature set with the same model configuration
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Model with proper regularization
        model = RandomForestClassifier(
            n_estimators=500,      # Increased trees
            max_depth=12,          # Limited depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Evaluate each feature set
        scores_all = cross_val_score(model, X_all, y, cv=cv, scoring='balanced_accuracy')
        scores_15 = cross_val_score(model, X_15, y, cv=cv, scoring='balanced_accuracy')
        scores_10 = cross_val_score(model, X_10, y, cv=cv, scoring='balanced_accuracy')
        
        print(f"\nFeature Set Comparison (Balanced Accuracy):")
        print(f"  All features: {scores_all.mean():.4f} (+/- {scores_all.std() * 2:.4f})")
        print(f"  Top 15: {scores_15.mean():.4f} (+/- {scores_15.std() * 2:.4f})")
        print(f"  Top 10: {scores_10.mean():.4f} (+/- {scores_10.std() * 2:.4f})")
        
        # Select best feature set based on CV performance
        feature_sets = {
            'all': (X_all, selected_features_all, scores_all),
            '15': (X_15, selected_features_15, scores_15),
            '10': (X_10, selected_features_10, scores_10)
        }
        
        best_set = max(feature_sets.keys(), key=lambda k: feature_sets[k][2].mean())
        print(f"\nBest feature set: {best_set} features")
        
        return feature_sets[best_set]
    
    def final_model_training_and_evaluation(self, X_best, y, selected_features):
        """Train final model and evaluate with proper train/test split"""
        print("\n--- [5/8] Final model training and evaluation ---")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_best, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        
        # Train model with proper regularization
        final_model = RandomForestClassifier(
            n_estimators=500,      # Increased trees
            max_depth=12,          # Limited depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return final_model, X_train, X_test, y_train, y_test
    
    def permutation_significance_test(self, model, X_test, y_test):
        """Perform permutation test for statistical significance"""
        print("\n--- [6/8] Permutation significance test ---")
        
        # Permutation test for statistical significance
        score, permutation_scores, pvalue = permutation_test_score(
            model, X_test, y_test, scoring='balanced_accuracy', 
            n_permutations=1000, n_jobs=-1, random_state=42
        )
        
        print(f"Permutation Test Results:")
        print(f"  True score: {score:.4f}")
        print(f"  Permutation scores mean: {permutation_scores.mean():.4f}")
        print(f"  P-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("  Result is statistically significant (p < 0.05)")
        else:
            print("  Result is not statistically significant (p >= 0.05)")
            
        return score, pvalue
    
    def model_calibration_analysis(self, model, X_train, y_train, X_test, y_test):
        """Analyze model calibration"""
        print("\n--- [7/8] Model calibration analysis ---")
        
        # Calibrate the model
        cal_clf = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        cal_clf.fit(X_train, y_train)
        
        # Evaluate calibrated model
        y_pred_cal = cal_clf.predict(X_test)
        accuracy_cal = accuracy_score(y_test, y_pred_cal)
        balanced_acc_cal = balanced_accuracy_score(y_test, y_pred_cal)
        
        print(f"Calibrated Model Performance:")
        print(f"  Accuracy: {accuracy_cal:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc_cal:.4f}")
        
        return cal_clf
    
    def feature_importance_analysis(self, model, X_train, selected_features):
        """Analyze feature importance"""
        print("\n--- [8/8] Feature importance analysis ---")
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 10 Most Important Features:")
        for i in range(min(10, len(selected_features))):
            print(f"  {i+1}. {selected_features[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importances.png'))
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete improved analysis pipeline"""
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Advanced preprocessing
        X_processed = self.advanced_preprocessing(X)
        
        # Nested cross-validation
        nested_scores, nested_scores_acc = self.nested_cross_validation(X_processed, y)
        
        # Feature selection comparison
        X_best, selected_features, cv_scores = self.feature_selection_comparison(X_processed, y)
        
        # Final model training and evaluation
        final_model, X_train, X_test, y_train, y_test = self.final_model_training_and_evaluation(
            X_best, y, selected_features)
        
        # Permutation significance test
        score, pvalue = self.permutation_significance_test(final_model, X_test, y_test)
        
        # Model calibration analysis
        calibrated_model = self.model_calibration_analysis(final_model, X_train, y_train, X_test, y_test)
        
        # Feature importance analysis
        self.feature_importance_analysis(final_model, X_train, selected_features)
        
        # Confusion matrix
        y_pred = final_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=final_model.classes_, yticklabels=final_model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_improved.png'), bbox_inches='tight')
        plt.close()
        
        print("\n=== IMPROVED ANALYSIS COMPLETE ===")
        print(f"Nested CV Balanced Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f})")
        print(f"Test Set Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
        print(f"Statistical significance (p-value): {pvalue:.4f}")
        
        self.best_model = final_model
        return {
            'model': final_model,
            'calibrated_model': calibrated_model,
            'selected_features': selected_features,
            'nested_scores': nested_scores,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
        }

def main():
    """Main function to run the improved analysis"""
    classifier = ImprovedAnemiaClassifier()
    results = classifier.run_complete_analysis()
    return results

if __name__ == "__main__":
    results = main()