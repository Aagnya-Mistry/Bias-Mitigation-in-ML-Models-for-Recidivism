# Bias Mitigation in Machine Learning Models using Explainable AI for Recidivism

## Overview
This project explores **machine learning models** for predicting criminal recidivism (reoffending) using the **ProPublica COMPAS dataset**.  
The key focus is not only accuracy but also **fairness** — ensuring models do not discriminate by **race** or **sex**.  

We test multiple models, explain their predictions with **XAI tools (SHAP, Fairlearn)**, and apply **bias mitigation techniques** to improve equity.  

---

## 🔎 Research Flow

### 1. Dataset
- **ProPublica COMPAS dataset** – widely used for fairness studies in criminal justice.  
- Includes demographics, criminal history, charges, and outcomes.  
- Used because it highlights **real-world bias challenges** in AI.

### 2. Preprocessing
- Dropped irrelevant IDs and cleaned missing data.  
- **One-hot encoded** sex & race → for ML interpretability.  
- Why: Makes raw data usable for models and fairness analysis.  

### 3. Models
We compared **simple, margin-based, and ensemble models**:  
- **Logistic Regression (baseline)** – interpretable, sets a benchmark.  
- **Support Vector Classifier (SVC)** – captures complex boundaries.  
- **Gradient Boosted Classifier (GBC)** – strong with non-linear patterns.  
- **XGBoost (XGB)** – optimized boosting, handles large feature interactions.  
- Why: To see which models balance accuracy and fairness best.  

### 4. Explainability
- **SHAP (feature importance)** → Shows how priors, age, sex, race affect predictions.  
- **Fairlearn (group fairness)** → Measures disparities across groups.  
- Why: Makes models transparent & exposes hidden bias.  

### 5. Bias Mitigation
- **In-processing**: *Adversarial Debiasing* – forces models to predict without relying on sensitive features.  
- **Post-processing**: *ROC, Equalized Odds, CEO* – adjust outputs to reduce unfairness.  
- Why: To reduce discrimination without losing accuracy.  

### 6. Results
- **SVC = Best trade-off**:  
  - Accuracy ~68%  
  - Fairness (Disparate Impact) ≈ 1.0 (ideal)  
- Logistic Regression → high accuracy but high bias.  
- XGBoost → fairer but less accurate.  
- Why: Shows fairness interventions **work without hurting performance**.  

---

## Repository
- `compas-scores-two-years.csv` – dataset.  
- `Final_Processed_Propublica_Compas.ipynb` – preprocessing & EDA.  
- `Model_accuracies_on_is_recid.ipynb` – model training on `is_recid`.  
- `Model_accuracies_on_two_year_recid.ipynb` – model training on `two_year_recid`.  
- `Bias Mitigation final.pdf` – full research paper.  

---

## ⚙️ Usage
```bash
pip install pandas numpy scikit-learn xgboost shap fairlearn matplotlib seaborn
