import pandas as pd
import numpy as np

# Define the expected feature columns
FEATURE_COLUMNS = [
    'WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 
    'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT'
]

def array_to_dataframe(X, columns=None):
    """Convert numpy array to pandas DataFrame"""
    if columns is None:
        columns = FEATURE_COLUMNS
    return pd.DataFrame(X, columns=columns)

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
    return df_copy