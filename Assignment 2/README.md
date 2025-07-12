# 🦅 Week 2 – Data Wrangling and Classification Evaluation  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-11-10

---

## Overview
This assignment builds upon basic data wrangling techniques and introduces classification performance evaluation. You will:
- Perform filtering, selection, and mutation operations in R
- Work with real-world wildlife strike data
- Generate confusion matrices and compute model evaluation metrics such as:
  - Kappa
  - Matthew's Correlation Coefficient (MCC)
  - ROC curve

---

## Files Included
- `DSP-569_Week2Assignment - Randy Sprouse.R` – Full R script  
- `DSP-569_Week2Assignment---Randy-Sprouse.html` – Rendered HTML output  
- `simMat_dat.csv` – Sample data for classification evaluation

---

## Key Topics
- `dplyr` operations: `filter()`, `select()`, `mutate()`, `group_by()`, `summarize()`
- Handling missing values and factor levels
- Computing confusion matrix and deriving evaluation statistics
- Plotting and interpreting ROC curves

---

## Data Sources
- Wildlife strike data: [wildlife_impacts.csv](https://github.com/jhelvy/p4a/raw/main/data/wildlife_impacts.csv)  
- Simulated classification matrix data: `simMat_dat.csv`

---

## Summary of Tasks
- Filter aircraft strikes costing over \$500,000  
- Calculate state-level mean strike heights  
- Subset and clean data for modeling  
- Compute confusion matrix, Kappa, and MCC  
- Generate and interpret a ROC curve for model performance

---

## Tools Used
- **R Packages**: `tidyverse`, `caret`, `pROC`, `mltools`, `vcd`, `data.table`

---

## Notes
- Code is adapted from [Data Wrangling for Public Policy](https://p4a.jhelvy.com/data-wrangling)  
- Evaluation metrics are calculated both manually and with package functions
