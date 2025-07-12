# 🤖 Week 6 – Artificial Neural Networks (ANN)  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-12-08

---

## Overview
This assignment focuses on **regression using neural networks** to predict the **age of possums** based on morphometric data. We explore simple and complex ANN architectures, evaluate performance, and compare to other techniques like linear regression and KNN.

---

## Files Included
- `possum.csv` – Dataset of possum measurements  
- `DSP-569_Week6Assignment - Randy Sprouse.R` – R script  
- `DSP-569_Week6Assignment---Randy-Sprouse.html` – Rendered HTML output  

---

## Key Topics
- Feedforward artificial neural networks
- Regression analysis using `neuralnet`
- Model tuning: hidden layers, activation functions
- Performance evaluation via correlation
- Comparison with linear regression and KNN

---

## Summary of Tasks

### 🔹 Data Preparation
- Removed non-numeric features: `case`, `site`, `pop`, `sex`
- Removed rows with missing values
- Normalized all features to [0, 1] range
- Split data 80/20 into training/testing sets

---

### 🔹 ANN – Default Model
- **Architecture**: Single hidden layer with 1 neuron
- **Activation**: Default (logistic)
- **Correlation (test set)**: ~0.475  
  *Moderate fit given dataset size (104 rows)*

---

### 🔹 ANN – Improved Model
- **Architecture**: Two hidden layers with 6 neurons each
- **Activation**: Custom softplus function
- **Correlation (test set)**: ~0.197  
  *Performance decreased with added complexity*

---

### 🔹 Linear Regression
- Tried full model and single-predictor model
- **Best correlation**: ~0.41  
  *Worse than default ANN*

---

### 🔹 K-Nearest Neighbors (KNN)
- Used `k = 12`
- **Correlation**: ~0.61  
  *Best performance among all models*

---

## Final Comparison

| Model               | Correlation |
|--------------------|-------------|
| ANN (1 hidden node) | ~0.48       |
| ANN (6–6, softplus) | ~0.20       |
| Linear Regression   | ~0.41       |
| KNN (k=12)          | ~0.61 ✅    |

- KNN regression provided the **best correlation** on this small dataset.
- ANN models are powerful but may **underperform on small datasets** without tuning and more data.
- Simpler models (KNN, linear regression) may generalize better in low-data scenarios.

---

## Tools Used
- **R Packages**: `neuralnet`, `caret`, `dplyr`, `tidyr`, `FNN`

---

## Observations
- ANN models need **larger datasets** to avoid overfitting or underfitting.
- **KNN** worked surprisingly well for small, clean, numeric data.
- Feature normalization was essential for both ANN and KNN.
