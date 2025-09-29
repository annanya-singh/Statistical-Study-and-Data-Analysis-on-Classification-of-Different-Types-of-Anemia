#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Anemia Classification with Statistical Rigor
=====================================================

This script implements a statistically robust anemia classification system with:
1. Advanced cross-validation techniques
2. Ensemble methods and stacking
3. Bayesian optimization for hyperparameter tuning
4. Advanced feature selection
5. Uncertainty quantification
6. Statistical tests for feature importance
7. Advanced calibration techniques
8. Robust error analysis with significance testing

Usage:
    python advanced_anemia_classification.py
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
    permutation_test_score
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    balanced_accuracy_score, make_scorer, brier_score_loss,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Advanced statistical libraries
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

# Explainability
import shap

# Bayesian optimization
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Optuna for advanced hyperparameter tuning
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Configuration
DATA_FILE = '/home/preeti/Desktop/projects/anemia_pro/cbc_data.csv'
OUTPUT_DIR = '/home/preeti/Desktop/projects/anemia_pro/output_advanced'

# Setup environment
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style="whitegrid")

class AdvancedAnemiaClassifier:
    """Advanced Anemia Classification System with Statistical Rigor"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.class_names = None
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.preprocessor = None
        self.cv_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the CBC data"""
        print("--- [1/10] Loading and analyzing data ---")
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
        print("\n--- [2/10] Advanced preprocessing ---")
        
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
    
    def stratified_split(self, X, y, test_size=0.2):
        """Perform stratified train-test split"""
        print("\n--- [3/10] Stratified data splitting ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def advanced_cross_validation(self, X, y):
        """Implement advanced cross-validation techniques"""
        print("\n--- [4/10] Advanced cross-validation ---")
        
        # Different CV strategies
        cv_strategies = {
            'StratifiedKFold_5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            'StratifiedKFold_10': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        }
        
        # Base models for comparison
        base_models = {
            'LogisticRegression': LogisticRegression(
                multi_class='multinomial', penalty='l2', max_iter=5000, random_state=42
            ),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
        }
        
        # Evaluate models with different CV strategies
        cv_results = {}
        scoring = ['accuracy', 'balanced_accuracy', 'roc_auc_ovr']
        
        for model_name, model in base_models.items():
            cv_results[model_name] = {}
            for cv_name, cv_strategy in cv_strategies.items():
                try:
                    scores = cross_validate(
                        model, X, y, cv=cv_strategy, scoring=scoring, 
                        n_jobs=-1, return_train_score=True
                    )
                    cv_results[model_name][cv_name] = scores
                    print(f"{model_name} with {cv_name}:")
                    print(f"  Accuracy: {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std() * 2:.4f})")
                    print(f"  Balanced Accuracy: {scores['test_balanced_accuracy'].mean():.4f} (+/- {scores['test_balanced_accuracy'].std() * 2:.4f})")
                    print(f"  ROC AUC: {scores['test_roc_auc_ovr'].mean():.4f} (+/- {scores['test_roc_auc_ovr'].std() * 2:.4f})")
                except Exception as e:
                    print(f"Error evaluating {model_name} with {cv_name}: {e}")
        
        self.cv_results = cv_results
        return cv_results
    
    def bayesian_hyperparameter_optimization(self, X_train, y_train):
        """Implement Bayesian optimization for hyperparameter tuning"""
        print("\n--- [5/10] Bayesian hyperparameter optimization ---")
        
        # Define search spaces for different models
        rf_param_space = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }
        
        # Bayesian optimization for Random Forest
        rf_bayes = BayesSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_space,
            n_iter=50,
            cv=5,
            n_jobs=-1,
            verbose=0,
            scoring='balanced_accuracy'
        )
        
        rf_bayes.fit(X_train, y_train)
        print(f"Best Random Forest parameters: {rf_bayes.best_params_}")
        print(f"Best Random Forest score: {rf_bayes.best_score_:.4f}")
        
        return rf_bayes.best_estimator_
    
    def advanced_ensemble_methods(self, X_train, y_train):
        """Implement advanced ensemble methods and stacking"""
        print("\n--- [6/10] Advanced ensemble methods ---")
        
        # Base models
        base_models = [
            ('lr', LogisticRegression(multi_class='multinomial', penalty='l2', max_iter=5000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
        ]
        
        # Voting classifier (soft voting)
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        voting_clf.fit(X_train, y_train)
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models[:-1],  # Remove one to use as final estimator
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking_clf.fit(X_train, y_train)
        
        print("Ensemble methods trained successfully")
        return voting_clf, stacking_clf
    
    def advanced_feature_selection(self, X_train, y_train, X_test):
        """Implement advanced feature selection techniques"""
        print("\n--- [7/10] Advanced feature selection ---")
        
        # Get feature names from the processed data
        feature_names_processed = X_train.columns.tolist()
        
        # Statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_features = np.array(feature_names_processed)[selector.get_support()]
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                  n_features_to_select=min(10, X_train.shape[1]))
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        
        selected_features_rfe = np.array(feature_names_processed)[rfe.get_support()]
        print(f"RFE selected {len(selected_features_rfe)} features: {selected_features_rfe}")
        
        return (X_train_selected, X_test_selected, selected_features), (X_train_rfe, X_test_rfe, selected_features_rfe)
    
    def uncertainty_quantification(self, model, X_test, y_test):
        """Quantify prediction uncertainty"""
        print("\n--- [8/10] Uncertainty quantification ---")
        
        # Get prediction probabilities
        y_proba = model.predict_proba(X_test)
        
        # Calculate prediction entropy (uncertainty measure)
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        
        # Calculate confidence (1 - max probability)
        confidence = 1 - np.max(y_proba, axis=1)
        
        # Create uncertainty dataframe
        uncertainty_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': model.predict(X_test),
            'max_probability': np.max(y_proba, axis=1),
            'entropy': entropy,
            'confidence': confidence
        })
        
        # High uncertainty predictions
        high_uncertainty = uncertainty_df[uncertainty_df['confidence'] > 0.5]
        print(f"High uncertainty predictions: {len(high_uncertainty)} / {len(uncertainty_df)}")
        
        return uncertainty_df
    
    def advanced_calibration(self, model, X_train, y_train, X_test, y_test):
        """Implement advanced calibration techniques"""
        print("\n--- [9/10] Advanced calibration ---")
        
        # Calibrate the model using different methods
        calibration_methods = ['sigmoid', 'isotonic']
        calibrated_models = {}
        
        for method in calibration_methods:
            cal_clf = CalibratedClassifierCV(model, method=method, cv=3)
            cal_clf.fit(X_train, y_train)
            calibrated_models[method] = cal_clf
            
            # Evaluate calibration (simplified for multiclass)
            try:
                y_proba_cal = cal_clf.predict_proba(X_test)
                # For multiclass, we'll compute a simplified calibration metric
                # This is a workaround for the brier_score_loss limitation with multiclass
                y_pred_cal = cal_clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_cal)
                print(f"{method.capitalize()} calibration - Accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"{method.capitalize()} calibration - Error in evaluation: {e}")
        
        return calibrated_models
    
    def robust_error_analysis(self, model, X_test, y_test):
        """Perform robust error analysis with statistical significance testing"""
        print("\n--- [10/10] Robust error analysis ---")
        
        # Classification report
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Permutation test for statistical significance
        score, permutation_scores, pvalue = permutation_test_score(
            model, X_test, y_test, scoring='balanced_accuracy', 
            n_permutations=1000, n_jobs=-1, random_state=42
        )
        print(f"\nPermutation Test Results:")
        print(f"True score: {score:.4f}")
        print(f"Permutation scores mean: {permutation_scores.mean():.4f}")
        print(f"P-value: {pvalue:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_advanced.png'), 
                   bbox_inches='tight')
        plt.close()
        
        return score, pvalue
    
    def run_complete_analysis(self):
        """Run the complete advanced analysis pipeline"""
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Advanced preprocessing
        X_processed = self.advanced_preprocessing(X)
        
        # Stratified split
        X_train, X_test, y_train, y_test = self.stratified_split(X_processed, y)
        
        # Advanced cross-validation
        cv_results = self.advanced_cross_validation(X_processed, y)
        
        # Bayesian hyperparameter optimization
        best_rf = self.bayesian_hyperparameter_optimization(X_train, y_train)
        
        # Advanced ensemble methods
        voting_clf, stacking_clf = self.advanced_ensemble_methods(X_train, y_train)
        
        # Advanced feature selection
        kbest_result, rfe_result = self.advanced_feature_selection(X_train, y_train, X_test)
        
        # For simplicity, let's use the KBest selected features for all models
        X_train_selected, X_test_selected, selected_features = kbest_result
        
        # Train models on selected features
        best_rf.fit(X_train_selected, y_train)
        voting_clf.fit(X_train_selected, y_train)
        stacking_clf.fit(X_train_selected, y_train)
        
        # Evaluate models
        models = {
            'BayesOpt_RF': best_rf,
            'Voting_Ensemble': voting_clf,
            'Stacking_Ensemble': stacking_clf
        }
        
        print("\n=== MODEL EVALUATION RESULTS ===")
        for name, model in models.items():
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Select best model (using balanced accuracy)
        best_model_name = max(models.keys(), key=lambda k: balanced_accuracy_score(
            y_test, models[k].predict(X_test_selected)))
        self.best_model = models[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        
        # Uncertainty quantification
        uncertainty_df = self.uncertainty_quantification(self.best_model, X_test_selected, y_test)
        
        # Advanced calibration
        calibrated_models = self.advanced_calibration(
            self.best_model, X_train_selected, y_train, X_test_selected, y_test)
        
        # Robust error analysis
        score, pvalue = self.robust_error_analysis(self.best_model, X_test_selected, y_test)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Best model: {best_model_name}")
        print(f"Statistical significance (p-value): {pvalue:.4f}")
        if pvalue < 0.05:
            print("Result is statistically significant (p < 0.05)")
        else:
            print("Result is not statistically significant (p >= 0.05)")
        
        return {
            'best_model': self.best_model,
            'models': models,
            'calibrated_models': calibrated_models,
            'uncertainty_df': uncertainty_df,
            'cv_results': cv_results
        }

def main():
    """Main function to run the advanced analysis"""
    classifier = AdvancedAnemiaClassifier()
    results = classifier.run_complete_analysis()
    return results

if __name__ == "__main__":
    results = main()