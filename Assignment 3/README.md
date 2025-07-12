# 📊 Week 3 – Multiple Linear and Logistic Regression  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-11-17

---

## Overview
This assignment applies both **multiple linear regression** and **logistic regression** to real-world biological data. You'll:
- Build and optimize a multiple linear regression model to predict human height
- Run logistic regression on fish maturity data
- Use `caret` and `Metrics` packages for model evaluation
- Interpret model performance using adjusted R², AIC, MAE, MSE, RMSE, and confusion matrices

---

## Files Included
- `GaltonHeightData.txt` – Data for height regression  
- `YERockfish.csv` – Fish maturity dataset  
- `DSP-569_Week3Assignment - Randy Sprouse.R` – Annotated R script  
- `DSP-569_Week3Assignment---Randy-Sprouse.html` – Rendered HTML output

---

## Key Topics
- Multiple linear regression modeling and diagnostics
- Stepwise model selection using `stepAIC()`
- Logistic regression for binary classification
- Performance evaluation:
  - Adjusted R² and AIC
  - MAE, MSE, RMSE
  - Confusion matrix (accuracy, sensitivity, specificity)

---

## Summary of Tasks

### 🔹 Multiple Linear Regression
- Load and clean Galton height data
- Split into training/test (80/20)
- Fit full model and perform feature selection with `stepAIC`
- Interpret coefficients and adjusted R²
- Predict height on test data and compute:
  - MAE
  - MSE
  - RMSE

### 🔹 Logistic Regression
- Load and clean rockfish maturity data
- Fit logistic regression with `age` and `length`
- Compare models with different feature orders
- Make predictions and generate confusion matrix
- Interpret feature significance and model accuracy

---

## Tools Used
- **R Packages**: `MASS`, `caret`, `Metrics`

---

## Notes
- Model optimization shows that some predictors (e.g. "Kids") may not improve model fit
- Logistic regression reveals that **length** is a stronger predictor of fish maturity than **age**
- Code is fully annotated with interpretation throughout

