# Bias Mitigation in Machine Learning Models for Recidivism Prediction

## Project Overview
This repository accompanies the research paper **“Bias Mitigation in Machine Learning Models for Recidivism Using Explainable AI”**.  
The work investigates how machine learning models used in criminal recidivism prediction (likelihood of re-offense) can unintentionally propagate bias based on **race** and **gender**.  

We compare multiple models on the **ProPublica COMPAS dataset**, analyze their fairness using **Explainable AI (SHAP, Fairlearn)**, and apply **bias mitigation techniques** (in-processing and post-processing).  

The central finding: **Support Vector Classifier (SVC)** offered the best trade-off between **accuracy** and **fairness**, demonstrating the potential for more equitable AI in criminal justice.

---

## Research Workflow

### 1. **Dataset**
- **Source**: ProPublica’s COMPAS dataset (Broward County, Florida defendants).  
- **Features**: Demographics (sex, race, age), criminal history, current charges, recidivism outcomes.  
- **Targets**:  
  - `is_recid` → Whether a person reoffended at all.  
  - `two_year_recid` → Whether a person reoffended within two years.  

### 2. **Preprocessing**
- Removed irrelevant identifiers (name, ID, dates, charge descriptions).  
- Encoded categorical features (sex, race) using **one-hot encoding**.  
- Cleaned missing values, duplicates, and inconsistencies.  
- Balanced consideration of `is_recid` and `two_year_recid`.  

### 3. **Exploratory Data Analysis**
- Correlation heatmaps showed **priors count**, **age**, and **sex** as strong predictors.  
- Found high overlap between `is_recid` and `two_year_recid`.  
- Rejected `is_violent_recid` as a target due to severe class imbalance.  

### 4. **Models Tested**
- **Logistic Regression (LogReg)** – baseline linear classifier.  
- **Support Vector Classifier (SVC)** – margin-based, kernel-enabled classifier.  
- **Gradient Boosted Classifier (GBC)** – sequential tree-based ensemble.  
- **XGBoost (XGB)** – optimized boosting algorithm.  

🔧 All models were trained with **hyperparameter tuning** (via `RandomizedSearchCV`) to improve performance.

### 5. **Explainability (XAI)**
- **SHAP**: Identified **priors count, age, sex, race** as dominant predictors.  
- **Findings**:  
  - More prior offenses → higher chance of predicted recidivism.  
  - Younger individuals (18–35) → higher predicted risk.  
  - Men and African-Americans were disproportionately associated with higher predicted recidivism.

### 6. **Fairness Assessment**
- Used **Fairlearn** to analyze group disparities.  
- Metrics: **Disparate Impact (DI)** and **group-wise accuracies**.  
- Results: Even with similar overall accuracy, models produced **unequal performance** across groups.

### 7. **Bias Mitigation Techniques**
- **In-processing**:  
  - *Adversarial Debiasing* – adversary model prevents predictions from encoding sensitive attributes.  
- **Post-processing**:  
  - *Reject Option Classification (ROC)*  
  - *Equalized Odds (EO)*  
  - *Calibrated Equalized Odds (CEO)*  
- **Combination**: In-processing + Post-processing for robust fairness.

### 8. **Results**
- **Support Vector Classifier (SVC)** consistently delivered the best balance of fairness and accuracy:  
  - Accuracy ≈ **68%**  
  - DI ≈ **1.0** (close to ideal fairness)  
- Logistic Regression had high accuracy but high bias.  
- XGBoost was fairer but slightly less accurate.  
- GBC was competitive but showed higher bias.  

**Key Finding**: *Bias mitigation methods significantly reduced unfairness without sacrificing accuracy, especially with SVC.*  

---

## Repository Structure
| File | Description |
|------|-------------|
| `compas-scores-two-years.csv` | Processed dataset from ProPublica. |
| `Final_Processed_Propublica_Compas.ipynb` | Data preprocessing, cleaning, and EDA. |
| `Model_accuracies_on_is_recid.ipynb` | Model training & fairness evaluation for `is_recid`. |
| `Model_accuracies_on_two_year_recid.ipynb` | Model training & fairness evaluation for `two_year_recid`. |
| `Bias Mitigation final.pdf` | Full research paper (methodology, results, discussion). |
| `README.md` | Project documentation (this file). |

---

## ⚙️ Installation & Usage

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost shap fairlearn matplotlib seaborn
