# 📧 Email Marketing Campaign Success Prediction (R)

[![R](https://img.shields.io/badge/Built%20With-R-blue?logo=r)](https://www.r-project.org/)
[![caret](https://img.shields.io/badge/ML-caret-orange?logo=rstudio)](https://topepo.github.io/caret/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📘 Project Overview

This project implements a **complete supervised learning workflow in R** to predict the success of a **skin care clinic’s email marketing campaign**.  
It includes full **data preparation, model training, ROC-AUC evaluation**, and **cross-validated neural network tuning** using the **caret** package.

**Key Features:**
- ✅ Stratified **Train/Test Split (80/20)**  
- ✅ **Decision Tree, Random Forest, Neural Network** comparison  
- ✅ **Cross-Validation (5×3 repeated)** with threshold tuning  
- ✅ **ROC-AUC Evaluation** and Confusion Matrices  
- ✅ **No data leakage** (scaling inside CV folds only)  
- ✅ **Regularized Neural Network** using `decay` to prevent overfitting  

**Dataset:** 683 customer records with demographics, purchase recency, billing history, and email engagement outcomes.

---

## 🎯 Business Problem

The clinic seeks to:
- Identify customers most likely to **open marketing emails**  
- Optimize targeting and reduce **wasted marketing spend**  
- Understand which features drive **email engagement**  
- Improve campaign **ROI** through predictive modeling  

---

## 📊 Dataset Description

| Variable           | Type      | Description                                        |
|--------------------|-----------|----------------------------------------------------|
| `Success`          | Binary    | Email opened (`1`) or not (`0`) — **Target**       |
| `Gender`           | Categorical | 1 = Male, 2 = Female                            |
| `AGE`              | Categorical | Age group: ≤30, ≤45, ≤55, >55                   |
| `Recency_Service`  | Numeric   | Days since last service purchase                   |
| `Recency_Product`  | Numeric   | Days since last product purchase                   |
| `Bill_Service`     | Numeric   | Service billing (last 3 months)                    |
| `Bill_Product`     | Numeric   | Product billing (last 3 months)                    |

**Response Rate:** ~28% email open rate  
**Split:** 80% training / 20% testing (stratified)

---

## 🔬 Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Checked structure, missing values, and class balance  
- Inspected variable distributions and correlations  
- Visualized recency and billing patterns vs. `Success`  

### 2️⃣ Data Preparation
- Converted categorical variables to factors  
- Removed non-predictive identifiers (`SN`)  
- Applied **center & scale** transformations (within CV only)  
- Stratified 80/20 split  

### 3️⃣ Models Implemented
| Model | Implementation | Key Feature |
|--------|----------------|-------------|
| Decision Tree | `rpart` | Baseline interpretability |
| Random Forest | `randomForest` | Ensemble robustness |
| Neural Network | `caret::avNNet` | CV tuning + regularization |

### 4️⃣ Neural Network Setup
- Hidden layer sizes: **1–3 neurons**  
- Regularization (`decay`): **0.001, 0.01, 0.1**  
- Cross-validation: **5-fold × 3 repeats**  
- Threshold selected from **out-of-fold CV predictions** (no test leakage)  

### 5️⃣ Evaluation Metrics
- **ROC-AUC** (area under curve)  
- **Accuracy**, **Sensitivity**, **Specificity**  
- **Confusion Matrix** per model  
- Combined ROC curve plot (DT vs RF vs NN)

---

## 📈 Model Performance Results

| Model | AUC | Accuracy | Sensitivity | Specificity |
|--------|-----|-----------|-------------|-------------|
| Decision Tree | ~0.80 | ~0.78 | ~0.84 | ~0.60 |
| Random Forest | ~0.85 | ~0.82 | ~0.88 | ~0.70 |
| Neural Net (CV-Tuned) | **~0.88** | **~0.83** | **~0.90** | **~0.74** |

> **Best Model:** Neural Network (CV-Tuned)  
> • Chosen for its balanced AUC, Sensitivity, and generalization  
> • Avoided overfitting through small hidden size and decay regularization  
> • Threshold derived from CV predictions, not test data  

---

## 📂 Project Structure

```
email-campaign-prediction-R/
├── data/
│   └── Email Campaign.csv
├── scripts/
│   └── email_campaign_modeling.R
├── models/
│   ├── decision_tree_model.rds
│   ├── random_forest_model.rds
│   ├── nn_cvtuned_model.rds
│   └── model_comparison_results.csv
├── reports/
│   ├── figures/
│   │   ├── decision_tree_plot.png
│   │   ├── variable_importance_rf.png
│   │   ├── roc_curves_all_models.png
│   │   └── confusion_matrices.png
│   └── performance_summary.html
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- **R 4.2+**  
- **RStudio**  
- Required libraries:
  ```r
  install.packages(c("caret","pROC","rpart","rpart.plot","randomForest","ggplot2","dplyr"))
  ```

### Run the Analysis
```r
source("scripts/email_campaign_modeling.R")
```

This will:
- ✅ Load and clean the dataset  
- ✅ Train Decision Tree, Random Forest, and NN  
- ✅ Perform 5×3 cross-validation for NN  
- ✅ Compute ROC-AUC for each model  
- ✅ Generate ROC plots and confusion matrices  
- ✅ Print a comparison table and identify the best model  

---

## 📊 Visualizations Generated

| Plot | Description |
|------|--------------|
| `decision_tree_plot.png` | Visual summary of the fitted Decision Tree |
| `variable_importance_rf.png` | Top variables ranked by Gini importance |
| `roc_curves_all_models.png` | ROC comparison of all models |
| `confusion_matrices.png` | Test set confusion matrices for each model |

---

## 💡 Insights & Recommendations

1. **Customer Engagement Drivers**
   - Higher service and product spending → increased open rates  
   - More recent interactions → higher likelihood to open emails  
   - Recency variables carry more predictive power than billing amounts  

2. **Marketing Actions**
   - Focus email campaigns on **recently active** and **high-spending** customers  
   - Use probability thresholds to segment customers for **A/B testing**  

3. **Technical Learnings**
   - Proper train/test split and cross-validation prevent overfitting  
   - Neural networks in R can perform competitively when tuned carefully  
   - Regularization (`decay`) and small network size are essential for small datasets  

---

## 📖 References
- Kuhn, M. (2008). *Building Predictive Models in R Using the caret Package*  
- Ripley, B. D. (1996). *Pattern Recognition and Neural Networks*  
- R Documentation: [caret](https://topepo.github.io/caret/), [avNNet](https://rdrr.io/cran/nnet/man/avNNet.html)

---

## 👨‍💼 Author
**William C. Phiri**  
📧 [wphiri@beda.ie]  
🔗 [LinkedIn](https://www.linkedin.com/in/william-phiri-866b8443/)  
🐙 [GitHub: Kochezz](https://github.com/kochezz)

---

## 📄 License
This project is licensed under the **MIT License** — see the LICENSE file for details.
