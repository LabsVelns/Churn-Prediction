# Customer Churn Prediction using CatBoost

## Overview

Customer churn is one of the most critical business problems for subscription-based and customer-centric organizations. Acquiring a new customer is often significantly more expensive than retaining an existing one, making early churn identification a high-impact business objective. Customer churn prediction is widely used to identify customers at risk of leaving and enable proactive retention strategies.

This project develops a machine learning pipeline to predict customer churn using structured customer data. The workflow covers:

* Exploratory Data Analysis (EDA)
* Data Cleaning & Preprocessing
* Missing Value Analysis
* Feature Selection
* Class Imbalance Handling
* Model Development using CatBoost
* Threshold Optimization
* Performance Evaluation

---

## Business Objective

The goal is to identify customers likely to churn before they leave.

### Business Benefits

* Reduce customer attrition
* Improve retention campaigns
* Optimize retention budgets
* Increase customer lifetime value
* Support data-driven decision making

---

## Dataset Summary

| Property             | Value          |
| -------------------- | -------------- |
| Records              | 50,000         |
| Features             | 231            |
| Numerical Features   | 191            |
| Categorical Features | 38             |
| Target Variable      | Customer Churn |

Target Distribution:

| Class         | Count  |
| ------------- | ------ |
| Retained (-1) | 46,328 |
| Churned (1)   | 3,672  |

### Key Observation

The dataset is highly imbalanced:

* Retained Customers: 92.66%
* Churned Customers: 7.34%

This makes Accuracy an unreliable metric since a model predicting every customer as retained would achieve approximately 92.6% accuracy while providing no business value.

---

# Exploratory Data Analysis

## Class Distribution

The target variable showed a significant imbalance between retained and churned customers.

This influenced:

* Validation strategy
* Evaluation metric selection
* Model training approach

---

## Missing Value Analysis

A detailed missing-value investigation revealed:

* Multiple features with 100% missing values
* Median missingness ≈ 97%
* Several features with more than 95% missing values

### Findings

Data quality emerged as the primary challenge rather than model selection.

Instead of blindly imputing all missing values, missingness patterns were analyzed to determine whether they carried predictive signal.

---

## Feature Cardinality Analysis

Unique value analysis was performed to identify:

* Constant features
* High-cardinality variables
* Potential identifier columns

Several features contained tens of thousands of unique values and were investigated for potential leakage risk.

---

## Missingness Correlation

Missing rates were compared across churn and retained classes.

Certain features displayed significantly different missingness patterns between churners and retained customers, indicating that missing values themselves may carry predictive information.

---

# Data Preprocessing

## Removed Features

### Fully Missing Columns

Columns containing 100% missing values were removed.

Reason:

These features contain no information and cannot contribute to prediction.

### Constant Columns

Columns containing only a single unique value were removed.

Reason:

Constant features contain no variance and therefore no predictive signal.

---

## Feature Type Identification

Features were separated into:

### Numerical Features

* float64
* int64

### Categorical Features

* object

Final Dataset:

| Feature Type | Count |
| ------------ | ----- |
| Numerical    | 173   |
| Categorical  | 34    |
| Total        | 207   |

---

## Missing Value Handling

### Numerical Variables

CatBoost's native missing value handling was utilized.

### Categorical Variables

Missing values were converted into:

```text
"Missing"
```

This allowed:

* Successful CatBoost training
* Preservation of missingness information
* Learning from missing-value patterns

---

# Validation Strategy

## Stratified K-Fold Cross Validation

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### Why Stratified K-Fold?

The dataset contains only 7.34% churners.

Stratification ensures that each fold preserves the original class distribution, leading to more reliable evaluation.

---

# Modelling Approach

## Model Selection

### CatBoostClassifier

CatBoost was selected because it:

* Handles categorical variables natively
* Handles missing values effectively
* Requires minimal preprocessing
* Performs strongly on tabular datasets
* Reduces feature engineering overhead

CatBoost is widely used for structured classification problems involving mixed numerical and categorical features.

---

## Baseline Configuration

```python
CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="F1",
    random_seed=42
)
```

---

# Handling Class Imbalance

## Approach 1: Class Weights

Class ratio:

```text
46328 / 3672 ≈ 12.6
```

Applied:

```python
class_weights=[1, 12.6]
```

Purpose:

Increase penalty for missing churners.

---

## Results (Weighted Model)

Confusion Matrix:

```text
[[6575 2690]
 [ 287  448]]
```

| Metric    | Score |
| --------- | ----- |
| Precision | 0.14  |
| Recall    | 0.61  |
| F1        | 0.23  |

### Observation

The model successfully captured more churners but generated a large number of false positives.

---

## Results (Without Class Weights)

Confusion Matrix:

```text
[[9262    3]
 [ 728    7]]
```

| Metric    | Score |
| --------- | ----- |
| Precision | 0.70  |
| Recall    | 0.01  |
| F1        | 0.02  |

### Observation

The model almost completely ignored the minority class.

This demonstrated the importance of imbalance handling.

---

# Threshold Optimization

Instead of relying solely on the default classification threshold (0.5), multiple thresholds were evaluated.

| Threshold | Precision | Recall | F1    |
| --------- | --------- | ------ | ----- |
| 0.30      | 0.091     | 0.937  | 0.165 |
| 0.40      | 0.110     | 0.805  | 0.194 |
| 0.50      | 0.143     | 0.610  | 0.231 |
| 0.60      | 0.193     | 0.386  | 0.258 |
| 0.70      | 0.277     | 0.192  | 0.227 |
| 0.80      | 0.410     | 0.044  | 0.079 |

### Best Observed Threshold

```text
Threshold = 0.60
F1 Score = 0.258
```

---

# Evaluation Metrics

Due to class imbalance, Accuracy was not used as the primary metric.

## Precision

Measures:

> Out of all predicted churners, how many actually churned?

---

## Recall

Measures:

> Out of all actual churners, how many were identified?

---

## F1 Score

Measures:

> Balance between Precision and Recall

Chosen as the primary optimization metric because business costs for false positives and false negatives were not explicitly defined.

---

## Future Metric

Recommended:

```text
PR-AUC (Precision Recall AUC)
```

PR-AUC is often more informative than Accuracy for highly imbalanced classification problems because it focuses on minority-class detection performance.

---

# Key Findings

* Data quality was a larger challenge than model selection.
* Most features contained extremely high missingness.
* Class imbalance significantly affected model behavior.
* CatBoost effectively handled mixed feature types.
* Class weighting improved churn detection.
* Threshold tuning improved F1 score.
* Additional gains are likely to come from feature engineering rather than hyperparameter tuning.

---

# Future Improvements

* Feature Importance Analysis
* SHAP Explainability
* PR-AUC Optimization
* Hyperparameter Tuning
* Feature Selection
* Missingness Indicator Features
* Comparison with:

  * XGBoost
  * LightGBM
  * Random Forest
* SMOTE / ADASYN Evaluation
* Cost-Sensitive Learning

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* Matplotlib
* Seaborn

---

# Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── reports/
│   └── technical_report.pdf
│
├── requirements.txt
├
└── README.md 
```

---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow for customer churn prediction, including exploratory analysis, preprocessing, imbalance handling, model development, threshold optimization, and business-oriented evaluation. The findings highlight that data quality and feature relevance are often more critical to success than model complexity, reinforcing the importance of thorough exploratory analysis before extensive model tuning.
