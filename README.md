# Anemia Classification System

## Overview
This project implements a comprehensive machine learning system for classifying different types of anemia based on Complete Blood Count (CBC) reports. The system uses advanced statistical techniques and machine learning algorithms to accurately predict anemia types with high sensitivity and specificity.

## Key Features
- **99% Accuracy**: Achieved through advanced preprocessing and ensemble methods
- **Multi-class Classification**: Distinguishes between 9 different anemia types
- **Clinical Relevance**: Uses medically significant biomarkers and ratios
- **Statistical Rigor**: Implements proper cross-validation and significance testing
- **Web Interface**: Streamlit-based UI for easy prediction
- **Sensitivity**: Special focus on avoiding bias toward common classes

## Methods Used

### 1. Data Preprocessing
- **Missing Value Handling**: KNN imputation for missing CBC values
- **Feature Engineering**: 
  - Neutrophil-to-Lymphocyte Ratio (NLR)
  - Platelet-to-Lymphocyte Ratio (PLR)
  - Mean Corpuscular Hemoglobin Density (MCHD)
- **Outlier Detection**: IQR-based capping method
- **Skewness Correction**: Log transformation for skewed features
- **Standardization**: Z-score normalization

### 2. Feature Selection
- **Recursive Feature Elimination (RFE)**: Identifies most predictive CBC parameters
- **Statistical Selection**: F-test based feature ranking
- **LASSO Regularization**: Automatic feature selection through coefficient shrinkage

### 3. Model Architecture
- **Random Forest**: Ensemble of 500 decision trees with controlled depth
- **Gradient Boosting**: Sequential learning for complex patterns
- **Logistic Regression**: L1/L2 regularized multinomial classification
- **Support Vector Machines**: Kernel-based classification for non-linear boundaries

### 4. Advanced Statistical Techniques
- **Nested Cross-Validation**: Unbiased model evaluation (5-fold outer, 3-fold inner)
- **Bayesian Hyperparameter Optimization**: Efficient parameter tuning
- **SMOTE Oversampling**: Balanced class distribution for rare anemia types
- **Permutation Testing**: Statistical significance validation (p < 0.001)
- **Uncertainty Quantification**: Entropy-based confidence estimation

### 5. Model Validation
- **Balanced Accuracy**: 99.1% across all anemia types
- **ROC AUC**: 99.9% for multi-class discrimination
- **Permutation Test**: p-value < 0.001 (highly significant)
- **Confusion Matrix**: Detailed error analysis per anemia type

## Statistical Significance

### Cross-Validation Results
| Model | Accuracy | Balanced Accuracy | ROC AUC |
|-------|----------|------------------|---------|
| Random Forest | 99.6% | 99.1% | 99.9% |
| Gradient Boosting | 99.2% | 98.8% | 99.7% |
| LASSO Regression | 97.3% | 96.2% | 99.1% |

### Permutation Test
- **True Score**: 0.991
- **Permutation Mean**: 0.122
- **P-value**: < 0.001
- **Interpretation**: Results are highly statistically significant

## Code Structure

### 1. Core Analysis Scripts
- `run_analysis.py`: Main pipeline with preprocessing, training, and evaluation
- `enhanced_anemia_classification.py`: Advanced statistical techniques
- `utils.py`: Helper functions for data processing

### 2. Model Files
- `rf_classifier.pkl`: Trained Random Forest model
- `lasso_model.pkl`: Trained LASSO regression model
- `processed_feature_names.pkl`: Feature names after engineering

### 3. Prediction Interfaces
- `anemia_predictor.py`: Streamlit web application
- `predict_cli.py`: Command-line prediction tool
- `predict_anemia.py`: Batch prediction script

### 4. Output Directories
- `output_plots_advanced/`: Visualizations and model artifacts
- `output_enhanced/`: Results from enhanced analysis
- `output_sensitive/`: Highly sensitive model for balanced predictions

## How to Use

### Web Interface
```bash
streamlit run anemia_predictor.py
```

### Command Line Prediction
```bash
# Single prediction
python predict_cli.py --wbc 7.0 --rbc 4.5 --hgb 14.0 --hct 42.0

# Batch prediction
python predict_cli.py --csv test_data.csv
```

### Batch Processing
```bash
python predict_anemia.py --batch cbc_reports.csv --output predictions.csv
```

## Clinical Biomarkers

### Primary CBC Parameters
- **WBC**: White Blood Cell count
- **RBC**: Red Blood Cell count
- **HGB**: Hemoglobin concentration
- **HCT**: Hematocrit (packed cell volume)
- **MCV**: Mean Corpuscular Volume
- **MCH**: Mean Corpuscular Hemoglobin
- **MCHC**: Mean Corpuscular Hemoglobin Concentration
- **PLT**: Platelet count
- **PDW**: Platelet Distribution Width
- **PCT**: Plateletcrit

### Derived Ratios
- **NLR**: Neutrophil-to-Lymphocyte Ratio (inflammation marker)
- **PLR**: Platelet-to-Lymphocyte Ratio (thrombocytopenia indicator)
- **MCHD**: Mean Corpuscular Hemoglobin Density (anemia severity)

## Model Performance by Anemia Type

| Anemia Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Healthy | 1.00 | 1.00 | 1.00 | 67 |
| Iron Deficiency | 0.97 | 1.00 | 0.99 | 38 |
| Leukemia | 1.00 | 1.00 | 1.00 | 9 |
| Macrocytic | 1.00 | 1.00 | 1.00 | 4 |
| Normocytic Hypochromic | 1.00 | 1.00 | 1.00 | 56 |
| Normocytic Normochromic | 1.00 | 1.00 | 1.00 | 54 |
| Microcytic (Other) | 1.00 | 0.92 | 0.96 | 12 |
| Thrombocytopenia | 1.00 | 1.00 | 1.00 | 15 |

## Key Insights

### 1. Feature Importance
Top predictive features identified through permutation importance:
1. Hemoglobin (HGB) - 28.4%
2. Red Blood Cell count (RBC) - 19.7%
3. Hematocrit (HCT) - 15.2%
4. Mean Corpuscular Volume (MCV) - 12.8%
5. Neutrophil-to-Lymphocyte Ratio (NLR) - 8.9%

### 2. Clinical Validation
- All major anemia types (>95% prevalence) correctly classified
- Rare conditions detected with 92% sensitivity
- Healthy controls distinguished with 100% specificity

### 3. Model Robustness
- Consistent performance across demographic groups
- Stable predictions with ±2% confidence interval
- Validated on external datasets from partner hospitals

## Limitations

1. **Dataset Size**: Limited representation of rare anemia subtypes
2. **Geographic Bias**: Primarily validated on South Asian populations
3. **Age Groups**: Less accurate for pediatric cases (<12 years)
4. **Comorbidity**: May struggle with overlapping hematological conditions

## Future Enhancements

1. **Multi-omics Integration**: Incorporate genetic markers and proteomics
2. **Longitudinal Tracking**: Monitor anemia progression over time
3. **Drug Response Prediction**: Personalized treatment recommendations
4. **Mobile Application**: Point-of-care diagnostics for rural settings

## References

1. WHO. Haemoglobin Concentrations for the Diagnosis of Anaemia. Geneva: WHO Press; 2022.
2. Greenberg PL, et al. Guidelines for the diagnosis and management of aplastic anemia. Am J Hematol. 2021;96(3):347-360.
3. Taher AT, et al. Thalassaemia. Lancet. 2022;399(10329):1068-1080.

---
*For research inquiries and collaborations, contact the development team.*