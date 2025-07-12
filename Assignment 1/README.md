# 🐧 Week 1 – Data Wrangling and PCA  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-11-03

---

## Overview
This assignment focuses on:
- Data wrangling and cleaning
- Principal Component Analysis (PCA)
- Correlation analysis
- Visualization using scree plots and biplots

Using two partial datasets of penguins, the goal is to combine them, clean the data, and explore patterns using PCA.

---

## Files Included
- `DSP-569_Week1Assignment - Randy Sprouse.R` – Full R script
- `DSP-569_Week1Assignment---Randy-Sprouse.html` – Rendered HTML output
- `penguins1.csv`, `penguins2.csv` – Raw datasets

---

## Key Steps
- Combine two datasets (`penguins1.csv` + `penguins2.csv`)
- Remove non-numeric and problematic columns (e.g., `sex`)
- Eliminate missing/invalid values
- Scale the data for PCA
- Generate a correlation matrix
- Run PCA and extract loadings
- Create scree plot and PCA biplot

---

## Summary of Results
- First two principal components explain **~96%** of variance
- Component 1 is driven by **culmen dimensions**
- Component 2 highlights contrast between **flipper length** and culmen size
- Dimensionality reduction is justified with minimal information loss

---

## Tools Used
- **R Packages**: `tidyverse`, `ggcorrplot`, `factoextra`, `flextable`

---

## Visuals
Output HTML includes correlation heatmap, scree plot, and biplot.
