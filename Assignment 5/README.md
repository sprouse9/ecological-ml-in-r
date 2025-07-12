# 🧠 Week 5 – Natural Language Processing (NLP)  
**DSP 569: Data Science Applications in Biology (Fall 2024)**  
**Randy Sprouse**  
📅 **Due Date:** 2024-12-01

---

## Overview
This assignment focuses on **Natural Language Processing (NLP)** for classification of biomedical abstracts. You'll perform preprocessing, vectorization (Bag-of-Words and TF-IDF), and train a **Support Vector Machine (SVM)** classifier to predict disease categories.

---

## Files Included
- `NLP_data2.csv` – Dataset of scientific abstracts and disease labels  
- `DSP-569_Week5Assignment - Randy Sprouse.R` – R script  
- `DSP-569_Week5Assignment---Randy-Sprouse.html` – Rendered HTML output  

---

## Key Topics
- Text preprocessing with `tm` package
- Bag-of-Words vs TF-IDF vectorization
- Support Vector Machine (SVM) classification
- Accuracy evaluation using confusion matrix

---

## Summary of Tasks

### 🔹 Preprocessing
- Sampled 1,000 random abstracts due to memory constraints
- Cleaned text by converting to lowercase, removing punctuation, numbers, stopwords, and whitespace
- Transformed cleaned text into a **Document-Term Matrix (DTM)**

### 🔹 Bag-of-Words (BoW) Model
- Used DTM to create training (80%) and testing (20%) sets
- Trained SVM model with linear kernel
- Evaluated predictions by converting probabilities to binary classes and computing accuracy

### 🔹 TF-IDF Model
- Applied TF-IDF weighting to the same DTM
- Re-trained SVM and evaluated performance
- Compared accuracy and confusion matrices with the BoW model

---

## Results

| Vectorization | Accuracy | Notes |
|---------------|----------|-------|
| Bag-of-Words  | ~83%     | Solid baseline performance |
| TF-IDF        | ~84%     | Slightly better; fewer false positives |

- TF-IDF improved the model by 1% in accuracy.
- Fewer **False Positives**, but slightly more **False Negatives**.
- Best model selection would depend on the application's tolerance for each type of error.

---

## Tools Used
- **R Packages**: `tm`, `slam`, `e1071`, `data.table`, `tidyverse`, `stringr`, `readr`

---

## Observations
- SVMs handle high-dimensional sparse text data well.
- TF-IDF helps reduce the influence of overly common words, improving generalization.
- Even basic NLP preprocessing yields strong predictive performance on biomedical text.
