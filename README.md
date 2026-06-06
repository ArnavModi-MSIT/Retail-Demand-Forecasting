# Retail Demand Forecasting & Inventory Optimization

Live Dashboard: https://arnavmodi-msit.github.io/Retail-Demand-Forecasting/

## Overview

This project develops an end-to-end Retail Demand Forecasting system using Machine Learning and Time-Series Feature Engineering. Historical sales patterns are transformed into demand forecasts, inventory recommendations, and business insights through an interactive analytics dashboard.

The system combines forecasting, explainability, and inventory optimization to support retail planning decisions.

---

## Problem Statement

Retail businesses often face:

- Stockouts caused by underestimating demand
- Overstocking due to inaccurate forecasting
- Inefficient inventory allocation across stores
- Limited visibility into drivers behind sales fluctuations

The objective is to forecast future demand accurately and convert predictions into actionable inventory recommendations.

---

## Dataset

The project uses retail sales data containing:

- Store information
- Department information
- Daily sales records
- Calendar variables
- Event indicators
- Product pricing information

### Dataset Scale

| Metric | Value |
|----------|----------|
| Records | 135,000+ |
| Features Engineered | 31 |
| Forecast Horizon | 28 Days |
| Stores | Multiple |
| Departments | Multiple |

---

## Machine Learning Pipeline

```text
Raw Sales Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Forecast Generation
        ↓
SHAP Explainability
        ↓
Inventory Optimization
        ↓
Interactive Dashboard
```

---

## Feature Engineering

### Lag Features

Historical sales information:

- Lag 1
- Lag 7
- Lag 14
- Lag 28

### Rolling Statistics

Demand trend indicators:

- Rolling Mean (7, 14, 30)
- Rolling Std (7, 30)

### Pricing Features

- Previous Price
- Price Change
- Percentage Price Change

### Calendar Features

- Month
- Quarter
- Weekday
- Weekend Indicator

### Event Features

- Event Type
- Event Name
- Event Flag

### Cyclical Features

- Month Sin/Cos
- Weekday Sin/Cos

---

## Models Evaluated

### Naive Baseline

Uses previous weekly sales as forecast.

### Random Forest Regressor

Tree-based ensemble model.

### LightGBM

Gradient boosting framework optimized for speed and efficiency.

### XGBoost

Gradient boosting model used as the final production model.

---

## Model Performance

| Model | RMSE | MAE | MAPE |
|---------|---------|---------|---------|
| XGBoost | 89.40 | 52.76 | 13.95% |
| LightGBM | 89.99 | 53.95 | 14.16% |
| Random Forest | 97.72 | 56.97 | 14.59% |
| Naive Baseline | 156.76 | 85.48 | 19.93% |

### Best Model

**XGBoost**

Performance improvement over Naive Baseline:

- RMSE reduced by approximately 43%
- MAPE reduced by approximately 30%

---

## Hyperparameter Optimization

Randomized Search Cross Validation was used to optimize:

- Learning Rate
- Maximum Depth
- Number of Estimators
- Subsample Ratio
- Column Sample Ratio
- Minimum Child Weight

---

## Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) was used to understand model predictions.

### Top Drivers

1. Rolling Mean 7
2. Lag 1
3. Lag 28
4. Lag 7
5. Lag 14

The analysis shows that recent demand history is the strongest predictor of future sales.

---

## Inventory Optimization

Forecasts are converted into inventory recommendations using:

```text
Recommended Inventory
=
Forecast Demand
+
Safety Stock
```

Risk categories:

- High Risk
- Medium Risk
- Low Risk

The system identified:

- 1,200+ high-risk store-department combinations
- Potential stockout scenarios
- Inventory planning opportunities

---

## Dashboard Features

### Overview

- Business KPIs
- Forecast summary
- Risk summary

### Forecast Analysis

- 28-Day Forecast Trend
- Demand visualization

### Model Performance

- RMSE Comparison
- MAE Comparison
- MAPE Comparison

### Explainability

- SHAP Summary Plot
- Mean SHAP Feature Importance

### Inventory Recommendations

- Risk Classification
- Safety Stock Analysis
- Recommended Inventory Levels

---

## Technology Stack

### Machine Learning

- Python
- XGBoost
- LightGBM
- Scikit-Learn

### Data Processing

- Pandas
- NumPy

### Explainability

- SHAP

### Visualization

- Chart.js
- HTML
- CSS
- JavaScript

### Deployment

- GitHub Pages

---

## Project Structure

```text
Retail-Demand-Forecasting/
│
├── dashboard/
│   ├── assets/
│   ├── css/
│   ├── data/
│   ├── js/
│   └── index.html
│
├── src/
│   ├── dataset.py
│   ├── eda.py
│   ├── features.py
│   ├── train.py
│   ├── future_forecast.py
│   ├── inventory.py
│   └── export_json.py
│
├── models/
│   └── xgb_model.pkl
│
├── outputs/
│   ├── model_results.csv
│   ├── future_forecasts.csv
│   ├── inventory_recommendations.csv
│   ├── shap_feature_importance.csv
│   └── feature_importance.csv
│
└── README.md
```

---

## Business Impact

This solution demonstrates how machine learning can be used to:

- Improve demand planning
- Reduce stockout risk
- Reduce excess inventory
- Improve inventory allocation
- Generate explainable forecasts for business stakeholders

---

## Author

**Arnav Modi**

B.Tech Information Technology  
Maharaja Surajmal Institute of Technology

LinkedIn: Add Your LinkedIn Link  
Portfolio: Add Your Portfolio Link
