# Insurance Premium Prediction - Capstone Project 1

## Overview
This capstone project focuses on building a predictive model to estimate insurance premium amounts based on customer demographics, financial information, health metrics, and insurance-related features. The project employs exploratory data analysis, data preprocessing, and machine learning techniques to develop an accurate premium prediction system.

## Project Objectives
1. **Understand the Dataset**: Perform comprehensive exploratory data analysis (EDA) to understand data distributions and relationships
2. **Data Cleaning**: Identify and handle missing values, outliers, and inconsistencies
3. **Feature Engineering**: Create meaningful features that improve model performance
4. **Model Development**: Build and evaluate multiple predictive models
5. **Optimization**: Fine-tune models for better accuracy and generalization

## Dataset Information

### Dataset Size
- **Total Records**: 278,860
- **Total Features**: 20
- **Missing Data**: Varies by feature (see details below)

### Feature Descriptions

#### Demographic Features
| Feature | Type | Missing | Range/Values | Description |
|---------|------|---------|--------------|-------------|
| Age | Numeric | 1.7% | 18-64 years | Customer age |
| Gender | Categorical | 0% | Male, Female | Customer gender |
| Marital Status | Categorical | 1.8% | Single, Married, Divorced | Relationship status |
| Number of Dependents | Numeric | 10% | 0-4 | Number of dependents |
| Education Level | Categorical | 0% | High School, Bachelor's, Master's, PhD | Education attainment |

#### Financial Features
| Feature | Type | Missing | Range/Values | Description |
|---------|------|---------|--------------|-------------|
| Annual Income | Numeric | 5.1% | $0-$149,997 | Yearly income (Mean: $42,089) |
| Credit Score | Numeric | 10.1% | 300-849 | Credit score (Mean: 574.36) |

#### Health & Lifestyle Features
| Feature | Type | Missing | Range/Values | Description |
|---------|------|---------|--------------|-------------|
| Health Score | Numeric | 3.8% | 0.04-93.88 | Health assessment score (Mean: 28.58) |
| Smoking Status | Categorical | 0% | Yes, No | Smoking habits |
| Exercise Frequency | Categorical | 0% | Daily, Weekly, Monthly, Rarely | Exercise habits |

#### Insurance & Vehicle Features
| Feature | Type | Missing | Range/Values | Description |
|---------|------|---------|--------------|-------------|
| Policy Type | Categorical | 0% | Basic, Comprehensive, Premium | Insurance coverage level |
| Previous Claims | Numeric | 29.3% | 0-9 | Number of prior claims (Mean: 0.998) |
| Insurance Duration | Numeric | 0% | 1-9 years | Policy duration (Mean: 5.01 years) |
| Vehicle Age | Numeric | 0% | 0-19 years | Age of vehicle (Mean: 9.52 years) |

#### Geographic & Other Features
| Feature | Type | Missing | Range/Values | Description |
|---------|------|---------|--------------|-------------|
| Location | Categorical | 0% | Urban, Suburban, Rural | Customer location type |
| Occupation | Categorical | 29.2% | Employed, Self-Employed, Unemployed | Employment type |
| Property Type | Categorical | 0% | House, Condo, Apartment | Property type |
| Customer Feedback | Categorical | 6.7% | Poor, Average, Good | Customer satisfaction |
| Policy Start Date | Date | 0% | Various dates | Policy initiation date |

#### Target Variable
| Feature | Type | Missing | Range | Description |
|---------|------|---------|-------|-------------|
| Premium Amount | Numeric | 0.7% | $0-$4,999 | **Target variable** (Mean: $966.12) |

## Exploratory Data Analysis (EDA)

### Key Findings

#### Distribution Analysis
- **Age Distribution**: Approximately uniform across age range 18-64, with mean age of 41 years
- **Income Distribution**: Right-skewed distribution with most customers earning under $75,000
- **Premium Distribution**: Right-skewed with median premium of $688 and mean of $966
- **Vehicle Age**: Average vehicle age is 9.5 years with range of 0-19 years
- **Insurance Duration**: Average policy duration is 5 years (range 1-9 years)

#### Categorical Distributions
- **Gender**: Slightly male-dominated (139,754 male vs 139,106 female)
- **Marital Status**: 
  - Single: 91,497 (32.8%)
  - Married: ~91,381 (32.8%)
  - Divorced: ~91,000 (32.6%)
- **Location**: 
  - Suburban: 93,482 (33.5%)
  - Urban: ~92,689 (33.2%)
  - Rural: ~92,689 (33.2%)
- **Policy Type**:
  - Premium: 93,298 (33.4%)
  - Comprehensive: ~92,781 (33.3%)
  - Basic: ~92,781 (33.2%)
- **Education Level**: PhD holders are most common (69,955 = 25.1%)
- **Smoking Status**: 50.1% smokers (139,635 yes vs 139,225 no)
- **Exercise Frequency**: Weekly (70,238 = 25.2%), Daily, Monthly, Rarely

#### Data Quality Issues
1. **Missing Values**:
   - Previous Claims: 29.3% missing (high)
   - Occupation: 29.2% missing (high)
   - Credit Score: 10.1% missing
   - Number of Dependents: 10% missing
   - Annual Income: 5.1% missing
   - Customer Feedback: 6.7% missing
   - Marital Status: 1.8% missing
   - Age: 1.7% missing
   - Premium Amount: 0.7% missing

2. **Outliers**: No extreme outliers detected in numeric features; values within reasonable ranges

### Statistical Summary
```
Age:               Mean=41.02,  Std=13.55,  Min=18,  Max=64
Annual Income:     Mean=42089,  Std=35445,  Min=0,   Max=149997
Number of Deps:    Mean=1.998,  Std=1.412,  Min=0,   Max=4
Health Score:      Mean=28.58,  Std=15.97,  Min=0.04, Max=93.88
Previous Claims:   Mean=0.998,  Std=1.001,  Min=0,   Max=9
Vehicle Age:       Mean=9.52,   Std=5.768,  Min=0,   Max=19
Credit Score:      Mean=574.36, Std=158.79, Min=300, Max=849
Insurance Duration: Mean=5.008, Std=2.581,  Min=1,   Max=9
Premium Amount:    Mean=966.12, Std=909.40, Min=0,   Max=4999
```

## Data Preprocessing Strategy

### 1. Missing Value Handling
- **Previous Claims & Occupation** (29% missing): Use K-Nearest Neighbors (KNN) imputation or mode-based imputation
- **Credit Score** (10% missing): Use median imputation or build separate predictive model
- **Number of Dependents** (10% missing): Use median imputation
- **Annual Income** (5% missing): Use median imputation by occupation/location
- **Customer Feedback** (6.7% missing): Use mode imputation
- **Age & Marital Status** (<2% missing): Use mode or median imputation
- **Premium Amount** (0.7% missing): Remove rows or use predictive imputation

### 2. Feature Engineering
- **Age Groups**: Create bins (18-25, 26-35, 36-45, 46-55, 56+)
- **Income Brackets**: Create salary bands
- **Policy Start Year**: Extract year from date feature
- **Policy Duration Groups**: Categorize duration into short/medium/long term
- **Risk Score**: Combine health score, smoking status, and previous claims
- **Combined Features**: Income-to-dependents ratio, vehicle-age-to-duration ratio

### 3. Encoding Categorical Variables
- **One-Hot Encoding**: For gender, location, property type, education level
- **Label Encoding**: For ordinal features (marital status, customer feedback)
- **Target Encoding**: For high-cardinality features (occupation) to prevent dimensionality explosion

### 4. Feature Scaling
- **Standardization**: Apply StandardScaler to numeric features for models sensitive to scale (KNN, SVM, Neural Networks)
- **Normalization**: Apply MinMaxScaler for tree-based models or when features need to be bounded [0,1]

### 5. Outlier Treatment
- **Insurance Duration & Vehicle Age**: Cap at 95th percentile if necessary
- **Premium Amount**: Review and handle zero premiums (possible data entry errors)
- **Health Score**: Verify extreme high/low values

## Proposed Modeling Approach

### Model Selection Strategy
1. **Baseline Models**:
   - Linear Regression (baseline comparison)
   - Decision Tree Regressor

2. **Intermediate Models**:
   - Random Forest Regressor
   - Gradient Boosting (XGBoost, LightGBM)
   - Support Vector Regression (SVR)

3. **Advanced Models**:
   - Neural Network (Multi-layer Perceptron)
   - Ensemble methods combining multiple models

### Model Evaluation Metrics
- **Mean Absolute Error (MAE)**: Average prediction error in dollar amount
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **R² Score**: Proportion of variance explained by the model
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### Cross-Validation Strategy
- **K-Fold Cross-Validation**: Use 5-fold CV to assess model stability
- **Stratified Splits**: Ensure train/test/validation sets have similar premium distributions

## Project Structure

```
Capstone-project-1/
├── README.md                          # This file
├── Capstone_project_1.ipynb           # Main notebook with analysis
├── data/
│   ├── insurance_data.csv            # Raw dataset
│   └── processed_data.csv            # Cleaned dataset (after preprocessing)
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Data_Cleaning.ipynb        # Data preprocessing & cleaning
│   ├── 03_Feature_Engineering.ipynb  # Feature creation & selection
│   └── 04_Modeling.ipynb             # Model training & evaluation
├── src/
│   ├── preprocessing.py              # Data cleaning functions
│   ├── feature_engineering.py        # Feature creation functions
│   └── models.py                     # Model definitions
└── results/
    ├── model_performance.csv         # Model comparison metrics
    ├── feature_importance.csv        # Feature importance rankings
    └── predictions.csv               # Final predictions on test set
```

## Installation & Dependencies

### Required Libraries
```python
import pandas as pd        # Data manipulation
import numpy as np         # Numerical computing
import matplotlib.pyplot   # Data visualization
import seaborn as sns      # Statistical data visualization
import scikit-learn        # Machine learning
import xgboost             # Gradient boosting
import lightgbm            # Light Gradient Boosting
```

### Installation Command
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

## Usage Instructions

1. **Data Loading**:
   ```python
   import pandas as pd
   df = pd.read_csv("data/insurance_data.csv")
   ```

2. **Exploratory Analysis**:
   - Run notebook `notebooks/01_EDA.ipynb` for comprehensive data exploration

3. **Data Preprocessing**:
   - Execute `notebooks/02_Data_Cleaning.ipynb` to handle missing values and outliers

4. **Feature Engineering**:
   - Use `notebooks/03_Feature_Engineering.ipynb` to create new features

5. **Model Development**:
   - Train models using `notebooks/04_Modeling.ipynb`

## Expected Outcomes

### Model Performance Targets
- **Linear Regression**: R² ≈ 0.60-0.70
- **Random Forest**: R² ≈ 0.75-0.85
- **Gradient Boosting**: R² ≈ 0.80-0.90
- **Neural Networks**: R² ≈ 0.80-0.92

### Prediction Accuracy
- Target MAE: < $150 (15% of mean premium)
- Target RMSE: < $250

## Key Insights & Business Applications

1. **Risk Assessment**: Premium predictions enable accurate risk pricing
2. **Customer Segmentation**: Identify high-risk vs. low-risk customer groups
3. **Policy Design**: Tailor coverage based on predicted premium exposure
4. **Fraud Detection**: Identify unusual premium patterns
5. **Competitive Pricing**: Set competitive rates based on customer profiles

## Next Steps

1. ✅ Complete EDA and data understanding (Current Phase)
2. ⬜ Implement missing value handling strategies
3. ⬜ Create and engineer features
4. ⬜ Train and evaluate baseline models
5. ⬜ Hyperparameter tuning for top models
6. ⬜ Final model selection and deployment preparation
7. ⬜ Generate predictions and business recommendations

## Contributors
- **Project Developer**: K121-hash
- **Project Type**: Capstone Project
- **Status**: In Progress

## License
This project is for educational purposes as part of a capstone assignment.

## Contact & Support
For questions or issues regarding this project, please refer to the GitHub repository issues section.

---

**Last Updated**: May 28, 2026
**Project Phase**: Exploratory Data Analysis (EDA) Complete
