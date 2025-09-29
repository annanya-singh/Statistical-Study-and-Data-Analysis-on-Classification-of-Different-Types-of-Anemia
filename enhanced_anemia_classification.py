#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Anemia Classification with Advanced Statistical Techniques
===================================================================

This script enhances the existing anemia classification system with:
1. Feature selection (RFE) to drop redundant CBC features
2. Controlled Random Forest tuning with proper depth limits
3. Nested cross-validation for fair model evaluation
4. Comparison of simpler models vs complex ensembles
5. Advanced feature selection techniques

Usage:
    python enhanced_anemia_classification.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    balanced_accuracy_score, accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Configuration
DATA_FILE = '/home/preeti/Desktop/projects/anemia_pro/cbc_data.csv'
OUTPUT_DIR = '/home/preeti/Desktop/projects/anemia_pro/output_enhanced'

# Setup environment
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style="whitegrid")

class EnhancedAnemiaClassifier:
    """Enhanced Anemia Classification System with Advanced Statistical Techniques"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.class_names = None
        
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
        """Advanced preprocessing with enhanced feature engineering"""
        print("\n--- [2/10] Advanced preprocessing with enhanced feature engineering ---")
        
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
            # Lymphocyte-to-Monocyte Ratio (approximated)
            df_copy['LMR'] = df_copy['LYMn'] / (df_copy['NEUTn'] + 1e-6)
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
    
    def advanced_feature_selection(self, X, y):
        """Implement advanced feature selection techniques (RFE/LASSO)"""
        print("\n--- [3/10] Advanced feature selection (RFE/LASSO) ---")
        
        # Get feature names
        feature_names = X.columns.tolist()
        print(f"Total features before selection: {len(feature_names)}")
        print("Features:", feature_names)
        
        # 1. Statistical Feature Selection (SelectKBest)
        selector_kbest = SelectKBest(score_func=f_classif, k=min(15, len(feature_names)))
        X_kbest = selector_kbest.fit_transform(X, y)
        selected_features_kbest = np.array(feature_names)[selector_kbest.get_support()]
        print(f"\nSelectKBest selected {len(selected_features_kbest)} features:")
        print(selected_features_kbest)
        
        # 2. Recursive Feature Elimination (RFE)
        # Using a simpler model for RFE to avoid overfitting
        rfe_estimator = RandomForestClassifier(
            n_estimators=100,      # Smaller for RFE
            max_depth=10,          # Controlled depth
            random_state=42
        )
        
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=min(12, len(feature_names)))
        X_rfe = rfe.fit_transform(X, y)
        selected_features_rfe = np.array(feature_names)[rfe.get_support()]
        print(f"\nRFE selected {len(selected_features_rfe)} features:")
        print(selected_features_rfe)
        
        # 3. Feature importance from a LASSO model for comparison
        # We'll use the LASSO model's coefficients as feature importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            multi_class='ovr',
            random_state=42,
            max_iter=1000
        )
        lasso.fit(X_scaled, y)
        
        # Get non-zero coefficients
        coef_importance = np.abs(lasso.coef_).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coef_importance
        }).sort_values('importance', ascending=False)
        
        # Select top features based on LASSO coefficients
        top_lasso_features = feature_importance_df[feature_importance_df['importance'] > 0]['feature'].values
        if len(top_lasso_features) > 12:
            top_lasso_features = feature_importance_df.head(12)['feature'].values
            
        print(f"\nLASSO selected {len(top_lasso_features)} features:")
        print(top_lasso_features)
        
        # Create transformed datasets
        X_lasso = X[top_lasso_features]
        
        return {
            'kbest': (X_kbest, selected_features_kbest),
            'rfe': (X_rfe, selected_features_rfe),
            'lasso': (X_lasso, top_lasso_features)
        }
    
    def nested_cross_validation(self, X, y, feature_method='rfe'):
        """Perform nested cross-validation to fairly evaluate models"""
        print(f"\n--- [4/10] Nested cross-validation with {feature_method} features ---")
        
        # Outer CV for model evaluation (5 splits)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Inner CV for hyperparameter tuning (3 splits)
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Define models to compare
        models = {
            'Simple_LR': LogisticRegression(
                multi_class='multinomial', 
                penalty='l2', 
                max_iter=5000, 
                random_state=42
            ),
            'LASSO_LR': LogisticRegression(
                multi_class='multinomial', 
                penalty='l1', 
                solver='saga',
                max_iter=5000, 
                random_state=42
            ),
            'RF_Controlled': RandomForestClassifier(random_state=42),
            'GB_Controlled': GradientBoostingClassifier(random_state=42)
        }
        
        # Hyperparameter grids with controlled parameters
        param_grids = {
            'Simple_LR': {
                'C': [0.1, 1.0, 10.0]
            },
            'LASSO_LR': {
                'C': [0.1, 1.0, 10.0]
            },
            'RF_Controlled': {
                'n_estimators': [300, 500],      # Increased trees
                'max_depth': [10, 12, 15],       # Controlled depth
                'min_samples_split': [5, 10],    # Min samples per split
                'min_samples_leaf': [2, 4],      # Min samples per leaf
                'max_features': ['sqrt', 'log2']
            },
            'GB_Controlled': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
        }
        
        # Evaluate each model with nested CV
        results = {}
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Grid search with inner CV
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=inner_cv, scoring='balanced_accuracy',
                n_jobs=-1, verbose=0
            )
            
            # Nested CV with outer CV
            nested_scores = cross_val_score(
                grid_search, X, y, cv=outer_cv,
                scoring='balanced_accuracy', n_jobs=-1
            )
            
            results[model_name] = {
                'scores': nested_scores,
                'mean': nested_scores.mean(),
                'std': nested_scores.std()
            }
            
            print(f"  {model_name}: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean'])
        print(f"\nBest model: {best_model_name} with {results[best_model_name]['mean']:.4f} balanced accuracy")
        
        return results, best_model_name
    
    def final_model_training(self, X, y, best_model_name, feature_method='rfe'):
        """Train the final model with best parameters"""
        print(f"\n--- [5/10] Final model training with {feature_method} features ---")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
        
        # Define the best model with controlled parameters
        if best_model_name == 'RF_Controlled':
            final_model = RandomForestClassifier(
                n_estimators=500,      # Increased trees
                max_depth=12,          # Controlled depth
                min_samples_split=5,   # Min samples per split
                min_samples_leaf=2,    # Min samples per leaf
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif best_model_name == 'GB_Controlled':
            final_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif best_model_name == 'LASSO_LR':
            final_model = LogisticRegression(
                multi_class='multinomial',
                penalty='l1',
                solver='saga',
                max_iter=5000,
                C=1.0,
                random_state=42
            )
        else:  # Simple_LR
            final_model = LogisticRegression(
                multi_class='multinomial',
                penalty='l2',
                max_iter=5000,
                C=1.0,
                random_state=42
            )
        
        # Train the model
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = final_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance ({best_model_name}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        return final_model, X_train, X_test, y_train, y_test
    
    def model_comparison(self, X, y):
        """Compare simpler models vs complex ensembles"""
        print("\n--- [6/10] Model comparison: Simple vs Complex ---")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Simple models
        simple_models = {
            'Logistic_Regression': LogisticRegression(
                multi_class='multinomial', 
                penalty='l2', 
                max_iter=5000, 
                random_state=42
            ),
            'LASSO_Regression': LogisticRegression(
                multi_class='multinomial', 
                penalty='l1', 
                solver='saga',
                max_iter=5000, 
                random_state=42
            ),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Complex models
        complex_models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Ensemble model
        ensemble_models = {
            'Voting_Ensemble': VotingClassifier(
                estimators=[
                    ('lr', simple_models['Logistic_Regression']),
                    ('rf', complex_models['Random_Forest']),
                    ('gb', complex_models['Gradient_Boosting'])
                ],
                voting='soft'
            )
        }
        
        # Evaluate all models
        all_models = {**simple_models, **complex_models, **ensemble_models}
        results = {}
        
        for model_name, model in all_models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc
            }
            
            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['balanced_accuracy'])
        print(f"\nBest overall model: {best_model_name}")
        
        return results, best_model_name
    
    def smote_implementation(self, X_train, y_train):
        """Apply SMOTE for handling imbalanced classes"""
        print("\n--- [7/10] Applying SMOTE for imbalanced classes ---")
        
        # Check class distribution
        class_counts = pd.Series(y_train).value_counts()
        print("Class distribution before SMOTE:")
        print(class_counts)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Check new distribution
        new_class_counts = pd.Series(y_train_balanced).value_counts()
        print("\nClass distribution after SMOTE:")
        print(new_class_counts)
        
        return X_train_balanced, y_train_balanced
    
    def permutation_significance_test(self, model, X_test, y_test):
        """Perform permutation test for statistical significance"""
        print("\n--- [8/10] Permutation significance test ---")
        
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
    
    def feature_importance_analysis(self, model, X_train, feature_names, method_name):
        """Analyze feature importance for the model"""
        print(f"\n--- [9/10] Feature importance analysis for {method_name} ---")
        
        # Different approaches based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("Top 10 Most Important Features:")
            for i in range(min(10, len(feature_names))):
                print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                # Multiclass - take mean of absolute coefficients
                coef_importance = np.abs(model.coef_).mean(axis=0)
            else:
                coef_importance = np.abs(model.coef_)
                
            indices = np.argsort(coef_importance)[::-1]
            
            print("Top 10 Most Important Features:")
            for i in range(min(10, len(feature_names))):
                print(f"  {i+1}. {feature_names[indices[i]]}: {coef_importance[indices[i]]:.4f}")
                
        else:
            print("Model does not support feature importance analysis")
    
    def final_evaluation(self, model, X_test, y_test, model_name):
        """Final comprehensive evaluation"""
        print(f"\n--- [10/10] Final evaluation of {model_name} ---")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        print(f"\nFinal Model Performance ({model_name}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png'), bbox_inches='tight')
        plt.close()
        
        return accuracy, balanced_acc
    
    def generate_visualizations(self, X_raw, y, model, X_test, y_test, feature_names, feature_method):
        """Generate comprehensive visualizations for the analysis"""
        print(f"\n--- Generating visualizations for {feature_method} ---")
        
        # 1. Class distribution visualization
        plt.figure(figsize=(10, 6))
        y.value_counts().plot(kind='bar')
        plt.title('Distribution of Anemia Types')
        plt.xlabel('Anemia Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
        plt.close()
        
        # 2. Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = X_raw.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation_heatmap.png'))
        plt.close()
        
        # 3. Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.title(f'Top 10 Feature Importances ({feature_method})')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_{feature_method}.png'))
            plt.close()
        
        # 4. Confusion matrix heatmap
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix ({feature_method})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{feature_method}.png'))
        plt.close()
        
        # 5. Model performance comparison visualization
        # This would be implemented if we had multiple models to compare
        
        print("Visualizations saved to output directory")
    
    def run_complete_analysis(self):
        """Run the complete enhanced analysis pipeline"""
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Advanced preprocessing
        X_processed = self.advanced_preprocessing(X)
        
        # Feature selection
        feature_sets = self.advanced_feature_selection(X_processed, y)
        
        # Compare feature selection methods
        print("\n=== FEATURE SELECTION COMPARISON ===")
        feature_comparison_results = {}
        
        for method_name, (X_selected, selected_features) in feature_sets.items():
            print(f"\nEvaluating with {method_name} features ({len(selected_features)} features)")
            
            # Nested CV for each feature set
            cv_results, best_model_name = self.nested_cross_validation(X_selected, y, method_name)
            
            feature_comparison_results[method_name] = {
                'cv_results': cv_results,
                'best_model': best_model_name,
                'selected_features': selected_features,
                'X_selected': X_selected
            }
            
            print(f"Best model for {method_name}: {best_model_name}")
            print(f"Performance: {cv_results[best_model_name]['mean']:.4f} (+/- {cv_results[best_model_name]['std'] * 2:.4f})")
        
        # Select best feature method
        best_feature_method = max(
            feature_comparison_results.keys(), 
            key=lambda k: feature_comparison_results[k]['cv_results'][feature_comparison_results[k]['best_model']]['mean']
        )
        
        print(f"\nBest feature selection method: {best_feature_method}")
        
        # Get the best feature set
        best_X = feature_sets[best_feature_method][0]
        best_features = feature_sets[best_feature_method][1]
        best_model_name = feature_comparison_results[best_feature_method]['best_model']
        
        # Final model training
        final_model, X_train, X_test, y_train, y_test = self.final_model_training(
            best_X, y, best_model_name, best_feature_method)
        
        # Apply SMOTE
        X_train_balanced, y_train_balanced = self.smote_implementation(X_train, y_train)
        
        # Retrain with balanced data if needed
        if len(np.unique(y_train)) != len(np.unique(y_train_balanced)):
            print("Retraining model with balanced data...")
            final_model.fit(X_train_balanced, y_train_balanced)
        
        # Model comparison
        model_comparison_results, overall_best_model = self.model_comparison(best_X, y)
        
        # Feature importance analysis
        self.feature_importance_analysis(final_model, X_train, best_features, best_feature_method)
        
        # Permutation significance test
        score, pvalue = self.permutation_significance_test(final_model, X_test, y_test)
        
        # Final evaluation
        accuracy, balanced_acc = self.final_evaluation(final_model, X_test, y_test, best_model_name)
        
        # Generate visualizations
        self.generate_visualizations(X_processed, y, final_model, X_test, y_test, best_features, best_feature_method)
        
        print("\n=== ENHANCED ANALYSIS COMPLETE ===")
        print(f"Best feature selection method: {best_feature_method}")
        print(f"Best model: {best_model_name}")
        print(f"Final Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Statistical significance (p-value): {pvalue:.4f}")
        
        self.best_model = final_model
        return {
            'model': final_model,
            'selected_features': best_features,
            'feature_method': best_feature_method,
            'best_model_name': best_model_name,
            'test_accuracy': accuracy,
            'test_balanced_accuracy': balanced_acc,
            'p_value': pvalue
        }

def main():
    """Main function to run the enhanced analysis"""
    classifier = EnhancedAnemiaClassifier()
    results = classifier.run_complete_analysis()
    return results

if __name__ == "__main__":
    results = main()