## Project Overview

This repository contains a high-performance machine learning pipeline designed to predict whether an accident victim was wearing a seatbelt (`Seatbelt_Used`) based on demographic and environmental factors. The project emphasizes **data integrity**, **feature discovery**, and **ensemble robustness**.

---

##  Technical Architecture

### 1. Robust Data Preprocessing
The pipeline addresses data quality issues through sophisticated imputation and scaling:
* **KNN Imputation:** Instead of simple mean/median, the model uses `KNNImputer` to estimate `Speed_of_Impact` by looking at the 5 most similar records based on `Age`, preserving local data patterns.
* **Stratified Splitting:** A 25% test split is created using **stratification**, ensuring the ratio of "Seatbelt Used" to "No Seatbelt" remains identical in both training and testing sets.
* **ColumnTransformer Pipeline:** 
    * **Numerical:** Median imputation followed by `StandardScaler` to bring all features to a common scale (mean=0, variance=1).
    * **Categorical:** Most-frequent imputation followed by `OneHotEncoder` with `handle_unknown='ignore'` to prevent crashes on novel test-set categories.

### 2. Advanced Feature Engineering
To improve predictive power, the dataset is enriched with derived variables:
* **Discretization:** `Age` and `Speed_of_Impact` are transformed from continuous values into ordinal categories (e.g., *Very_Young* to *Elderly*; *Very_Low* to *Extreme*) to capture non-linear risks.
* **Interaction Features:** 
    * `Helmet_Survived`: Combines safety gear usage and medical outcome.
    * `Age_Speed_Interaction`: Multiplies age by speed to capture the compounded risk factor.
* **Polynomial Terms:** Includes squared versions of `Age` and `Speed` to model parabolic relationships in safety behavior.

### 3. Class Imbalance Strategy
To ensure the model doesn't just "guess" the majority class, we employ **SMOTE (Synthetic Minority Over-sampling Technique)**:
* It generates synthetic examples for the underrepresented class in the training set.
* This forces the classifier to learn the decision boundary of the minority class rather than ignoring it.

---

##  Modeling & Hyperparameter Optimization

The project evaluates three primary models and one meta-model:

| Model | Key Configuration |
| :--- | :--- |
| **Random Forest** | 1000 trees, `class_weight='balanced'`, parallelized (n_jobs=-1). |
| **Gradient Boosting** | Learning rate of 0.01 with 1000 estimators for slow, robust learning. |
| **XGBoost** | Optimized with subsampling (0.8) and column sampling (0.8) to prevent overfitting. |
| **Stacking Classifier** | Uses the three models above as "base learners" and a **Logistic Regression** as the "final estimator" to weight their votes. |

### Hyperparameter Tuning
We use `GridSearchCV` with **Stratified K-Fold Cross-Validation (k=5)**. The search space includes:
* `max_depth`: Controlling tree complexity.
* `n_estimators`: Determining the number of boosting rounds.
* `learning_rate`: Balancing speed vs. accuracy in gradient descent.

---

## Evaluation Metrics & Visuals

The script generates four critical visualizations to validate the model:
1. **Confusion Matrix:** Breaks down True Positives (correctly predicted seatbelt) vs. False Positives (predicted seatbelt when none was used).
2. **ROC Curve & AUC:** The Area Under the Curve (AUC) score provides a single metric for the model's ability to distinguish between classes.
3. **Precision-Recall Curve:** Particularly useful for this imbalanced dataset, showing the trade-off between capturing all users (Recall) and ensuring they are actually users (Precision).
4. **Permutation Feature Importance:** A model-agnostic approach that shuffles features to see how much the accuracy drops, identifying the \"true\" drivers of the prediction.

---

## Requirements & Usage

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
