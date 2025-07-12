# 🌱 Week 4 – KNN, Decision Trees, and Random Forest  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-11-24

---

## Overview
This assignment applies three powerful classification techniques:
- **K-Nearest Neighbors (KNN)**
- **Decision Trees (C5.0)**
- **Random Forests**

You'll explore real-world biological and agricultural datasets, tune hyperparameters, and compare classifier performance using metrics like **accuracy**, **Kappa**, and **error rates**.

---

## Files Included
- `Pumpkin_Seeds_Dataset.csv` – Pumpkin seed morphology dataset  
- `mushrooms.csv` – Mushroom edibility classification dataset  
- `DSP-569_Week4Assignment - Randy Sprouse.R` – R script  
- `DSP-569_Week4Assignment---Randy-Sprouse.html` – Rendered HTML output

---

## Key Topics
- KNN normalization and tuning `k`
- Decision tree modeling with boosting (`C5.0`)
- Random forest ensemble modeling
- Model evaluation with:
  - CrossTables
  - Kappa statistic
  - Confusion matrices

---

## Summary of Tasks

### 🔹 K-Nearest Neighbors (KNN)
- Normalize pumpkin seed features
- Split data (75% train, 25% test)
- Train and evaluate model with `k = sqrt(n)`
- Investigate alternative `k` values (e.g., 1–45)
- Visualize number of classification errors vs. `k`

### 🔹 Decision Trees (C5.0)
- Clean mushroom dataset (remove `veil_type`)
- Split into train/test
- Train C5.0 model and evaluate with `CrossTable()` and `confusionMatrix()`
- Improve accuracy with boosting (`trials = 10`)
- Report and interpret Kappa statistics

### 🔹 Random Forest
- Train model on full mushroom dataset
- Predict outcomes and evaluate with Kappa
- Compare all three methods on classification performance

---

## Observations
- **KNN** achieved ~87% accuracy at `k = 15`, showing strong baseline performance
- **C5.0 Decision Tree** and **Boosted Tree** both achieved perfect classification (Kappa = 1)
- **Random Forest** also yielded perfect agreement (Kappa = 1), reinforcing its robustness in high-dimensional categorical data

---

## Tools Used
- **R Packages**: `class`, `C50`, `randomForest`, `gmodels`, `caret`, `OneR`, `tidyverse`

---

## Notes
- Boosting improved tree stability and accuracy
- Random forest matched boosted tree performance without needing additional tuning
- All models were reproducible using `set.seed()`
