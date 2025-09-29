# ==============================================================================
# SCRIPT: ANEMIA CLASSIFICATION (ADVANCED PREPROCESSING)
# ==============================================================================
#
# DESCRIPTION:
# This script implements a robust machine learning workflow for anemia
# classification, featuring an advanced data cleaning and preprocessing pipeline.
#
# KEY FEATURES:
#   1. Explicit analysis of missing data.
#   2. Feature Engineering (e.g., Neutrophil-to-Lymphocyte Ratio).
#   3. Outlier detection and capping using the IQR method.
#   4. Skewness detection and correction using log transformation.
#   5. All preprocessing steps are encapsulated in a scikit-learn Pipeline
#      to prevent data leakage and ensure reproducibility.
#   6. Model training, evaluation, explainability, and visualization.
#
# USAGE:
#   1. Install libraries from requirements.txt: pip install -r requirements.txt
#   2. Place your data file in the same directory.
#   3. Update the DATA_FILE variable below.
#   4. Run from the terminal: python run_analysis_advanced.py
#
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: SETUP & CONFIGURATION
# ------------------------------------------------------------------------------
print("--- [1/10] Setting up the environment ---")

# Core libraries
import pandas as pd
import numpy as np
import os
import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# SHAP
import shap

# --- Configuration ---
DATA_FILE = '/home/preeti/Desktop/projects/anemia_pro/cbc_data.csv'  # <-- IMPORTANT: CHANGE THIS FILENAME
OUTPUT_DIR = '/home/preeti/Desktop/projects/anemia_pro/output_plots_advanced'

# --- Setup Environment ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# SECTION 2: DATA LOADING AND INITIAL ANALYSIS
# ------------------------------------------------------------------------------
print(f"--- [2/10] Loading and analyzing data from '{DATA_FILE}' ---")
try:
    cbc_data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"FATAL ERROR: The file '{DATA_FILE}' was not found.")
    exit()

print(f"Successfully loaded data. Shape: {cbc_data.shape}")

# --- Missing Value Analysis ---
missing_values = cbc_data.isnull().sum()
missing_percent = (missing_values / len(cbc_data)) * 100
missing_df = pd.DataFrame({'count': missing_values, 'percent': missing_percent})
print("\nMissing Value Report:\n", missing_df[missing_df['count'] > 0])

# Prepare data for modeling
cbc_data['Diagnosis'] = cbc_data['Diagnosis'].astype('category')
X = cbc_data.drop('Diagnosis', axis=1)
y = cbc_data['Diagnosis']
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# ------------------------------------------------------------------------------
# SECTION 3: STRATIFIED DATA SPLITTING
# ------------------------------------------------------------------------------
print("\n--- [3/10] Performing stratified train-test split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


# ------------------------------------------------------------------------------
# SECTION 4: ADVANCED PREPROCESSING PIPELINE CONSTRUCTION
# ------------------------------------------------------------------------------
print("\n--- [4/10] Defining advanced preprocessing functions and pipeline ---")

# --- Custom Preprocessing Functions ---

def create_ratios(df):
    """ Feature Engineering: Create clinically relevant ratios. """
    df_copy = df.copy()
    # Neutrophil-to-Lymphocyte Ratio (NLR)
    # Add a small epsilon to avoid division by zero
    df_copy['NLR'] = df_copy['NEUTn'] / (df_copy['LYMn'] + 1e-6)
    return df_copy

def cap_outliers(df, iqr_factor=1.5):
    """ Outlier Handling: Cap outliers using the IQR method. """
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
    """ Skewness Correction: Apply log transform to highly skewed features. """
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=np.number).columns:
        if abs(df_copy[col].skew()) > skew_threshold:
            # Use log1p to handle zero values gracefully
            df_copy[col] = np.log1p(df_copy[col])
            print(f"   - Log-transformed '{col}' (skewness: {df_copy[col].skew():.2f})")
    return df_copy

# --- Build the Preprocessing Pipeline ---
# This pipeline chains all cleaning and transformation steps together.
# This is crucial for applying the same transformations to train and test data
# without data leakage.
#preprocessor = Pipeline(steps=[
   # ('imputer', KNNImputer(n_neighbors=5)),
    # Convert back to DataFrame for custom functions
    #('to_dataframe', FunctionTransformer(lambda x: pd.DataFrame(x, columns=numeric_features))),
    #('feature_engineering', FunctionTransformer(create_ratios)),
    #('outlier_capper', FunctionTransformer(cap_outliers)),
    #('skew_transformer', FunctionTransformer(log_transform_skewed)),
    #('scaler', StandardScaler())

print("Advanced preprocessing pipeline created successfully.")

# ------------------------------------------------------------------------------
# SECTION 5: MODEL TRAINING & HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------
print("\n--- [5/10] Training models with the advanced preprocessing pipeline ---")

# Define a named function to convert arrays to DataFrames (needed for pickling)
def array_to_dataframe(X, columns=None):
    """Convert numpy array to pandas DataFrame"""
    if columns is None:
        columns = numeric_features
    return pd.DataFrame(X, columns=columns)

# --- Model 1: LASSO Logistic Regression ---
# This is a standard sklearn Pipeline because it doesn't use SMOTE
lasso_full_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('to_dataframe', FunctionTransformer(array_to_dataframe, kw_args={'columns': numeric_features})),
    ('feature_engineering', FunctionTransformer(create_ratios)),
    ('outlier_capper', FunctionTransformer(cap_outliers)),
    ('skew_transformer', FunctionTransformer(log_transform_skewed)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', max_iter=5000, C=1.0, random_state=42))
])

# --- Model 2: Random Forest with SMOTE ---
# This MUST be an ImbPipeline, with all steps listed individually.
rf_full_pipeline = ImbPipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('to_dataframe', FunctionTransformer(array_to_dataframe, kw_args={'columns': numeric_features})),
    ('feature_engineering', FunctionTransformer(create_ratios)),
    ('outlier_capper', FunctionTransformer(cap_outliers)),
    ('skew_transformer', FunctionTransformer(log_transform_skewed)),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

# --- Hyperparameter Tuning for Random Forest ---
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(rf_full_pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
print(f"\nBest parameters for Random Forest: {grid_search_rf.best_params_}")

# --- Fit the final LASSO model ---
lasso_full_pipeline.fit(X_train, y_train)
print("Final LASSO model fitted successfully.")


# ------------------------------------------------------------------------------
# The rest of the script (Evaluation, Explainability, Plots) remains largely
# the same, but now operates on the properly preprocessed data via the pipelines.
# ------------------------------------------------------------------------------

# SECTION 6: MODEL EVALUATION
print("\n--- [6/10] Evaluating models on the test set ---")
# (Code is identical to the previous script, using `best_rf_model` and `lasso_full_pipeline`)
y_pred_rf = best_rf_model.predict(X_test)
y_proba_rf = best_rf_model.predict_proba(X_test)
y_pred_lasso = lasso_full_pipeline.predict(X_test)
y_proba_lasso = lasso_full_pipeline.predict_proba(X_test)

print("\n" + "="*50)
print("           TUNED RANDOM FOREST - EVALUATION")
print("="*50)
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC Score (OVO): {roc_auc_score(y_test, y_proba_rf, multi_class='ovo'):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_rf.png'), bbox_inches='tight')
plt.close()
print("Saved Random Forest confusion matrix plot.")

print("\n" + "="*50)
print("       LASSO LOGISTIC REGRESSION - EVALUATION")
print("="*50)
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_lasso):.4f}")
print(f"ROC AUC Score (OVO): {roc_auc_score(y_test, y_proba_lasso, multi_class='ovo'):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lasso))

cm_lasso = confusion_matrix(y_test, y_pred_lasso)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_lasso, annot=True, fmt='d', cmap='Blues', xticklabels=lasso_full_pipeline.classes_, yticklabels=lasso_full_pipeline.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LASSO Regression Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_lasso.png'), bbox_inches='tight')
plt.close()
print("Saved LASSO confusion matrix plot.")

# SECTION 7: MODEL EXPLAINABILITY
print("\n--- [7/10] Generating model explainability plots ---")

# --- Get Original and Processed Feature Names ---
# The original feature names are what permutation_importance uses.
original_feature_names = X_test.columns.tolist()

# The processed feature names (with 'NLR') are what SHAP will use.
temp_df_with_ratios = create_ratios(X_train.head(1))
processed_feature_names = temp_df_with_ratios.select_dtypes(include=np.number).columns.tolist()
print(f"Original feature count: {len(original_feature_names)}")
print(f"Processed feature count: {len(processed_feature_names)}")

# --- Permutation Importance ---
# This plot will use the ORIGINAL feature names.
perm_importance = permutation_importance(best_rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
# Create a DataFrame using the original feature names, which have the correct length.
perm_df = pd.DataFrame({'feature': original_feature_names, 'importance': perm_importance.importances_mean}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(perm_df['feature'], perm_df['importance'])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Random Forest - based on original features)")
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_permutation.png'), bbox_inches='tight')
plt.close()
print("Saved Permutation Importance plot.")


# --- SHAP Summary Plot ---
# This plot will use the PROCESSED feature names.
# Get the part of the pipeline that does the preprocessing (all steps except the last two)
preprocessor_for_shap = best_rf_model[:-2]

# Get the final model and the transformed test data
final_rf_classifier = best_rf_model.named_steps['model']
processed_X_test = preprocessor_for_shap.transform(X_test)

# Now, we create the explainer and SHAP values on the transformed data
explainer = shap.TreeExplainer(final_rf_classifier)
shap_values = explainer.shap_values(processed_X_test)

# When we plot, we provide the correctly-lengthed processed_feature_names
plt.figure()
shap.summary_plot(shap_values, pd.DataFrame(processed_X_test, columns=processed_feature_names), plot_type="bar", class_names=best_rf_model.classes_, show=False)
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_plot.png'), bbox_inches='tight')
plt.close()
print("Saved SHAP summary plot.")

# SECTION 8: EXPLORATORY DATA ANALYSIS VISUALIZATIONS
print("\n--- [8/10] Generating and saving EDA plots ---")
# (Code is identical to the previous script)
for col in numeric_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(cbc_data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'dist_{col}.png'), bbox_inches='tight')
    plt.close()
print("Saved feature distribution plots.")

for col in numeric_features:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=cbc_data, x='Diagnosis', y=col)
    plt.title(f'{col} by Diagnosis')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{col}_by_diagnosis.png'), bbox_inches='tight')
    plt.close()
print("Saved boxplots by diagnosis.")

plt.figure(figsize=(16, 12))
corr_matrix = cbc_data.select_dtypes(include=np.number).corr(method='spearman')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print("Saved correlation heatmap.")


# SECTION 9: PCA PLOT
print("\n--- [9/10] Generating and saving PCA plot on preprocessed data ---")
# Use the preprocessor part of the final pipeline to get cleaned data for PCA
# This ensures we use the exact same steps that the model was trained on
preprocessor_for_pca = best_rf_model[:-2] # All steps except smote and model
X_train_processed_for_pca = preprocessor_for_pca.fit_transform(X_train, y_train)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_train_processed_for_pca)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Diagnosis'] = y_train.values

plt.figure(figsize=(12, 9))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Diagnosis', s=80, alpha=0.8)
plt.title('PCA of Preprocessed CBC Data (Colored by Diagnosis)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_plot_processed.png'), bbox_inches='tight')
plt.close()
print("Saved PCA plot on processed data.")

# SECTION 10: MODEL SAVING
print("\n--- [10/10] Saving trained models for future use ---")
import joblib

# Create a simplified model for prediction that doesn't rely on custom functions
# We'll extract the trained Random Forest model and create a standalone pipeline

# Get the trained Random Forest classifier
trained_rf = best_rf_model.named_steps['model']

# Get feature names after preprocessing (including engineered features)
temp_df_with_ratios = create_ratios(X_train.head(1))
processed_feature_names = temp_df_with_ratios.select_dtypes(include=np.number).columns.tolist()

# Save the trained Random Forest model and feature names
joblib.dump(trained_rf, os.path.join(OUTPUT_DIR, 'rf_classifier.pkl'))
joblib.dump(processed_feature_names, os.path.join(OUTPUT_DIR, 'processed_feature_names.pkl'))

# Also save a sample of the preprocessed training data for reference
X_train_sample_processed = best_rf_model[:-2].transform(X_train.head(100))  # All steps except SMOTE and model
joblib.dump(X_train_sample_processed, os.path.join(OUTPUT_DIR, 'sample_preprocessed_data.pkl'))

# Save the original feature names
joblib.dump(numeric_features, os.path.join(OUTPUT_DIR, 'original_feature_names.pkl'))

print("Saved Random Forest classifier as 'rf_classifier.pkl'")
print("Saved processed feature names as 'processed_feature_names.pkl'")
print("Saved sample preprocessed data as 'sample_preprocessed_data.pkl'")
print("Saved original feature names as 'original_feature_names.pkl'")

print("\n--- Analysis complete. All outputs saved to 'output_plots_advanced' directory. ---")