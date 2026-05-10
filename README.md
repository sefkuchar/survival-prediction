# Accident Analysis: Seatbelt Usage Prediction Model

This project implements a comprehensive machine learning pipeline to predict seatbelt usage among accident victims based on demographic and situational data. It leverages advanced preprocessing, feature engineering, and ensemble modeling techniques to achieve high predictive accuracy.

---

## ## Project Overview

The primary goal is to analyze accident data to identify factors influencing seatbelt usage and build a robust classifier. The project addresses challenges such as missing values and class imbalance to provide reliable insights for safety recommendations.

---

## ## Technical Workflow

### ### 1. Advanced Preprocessing & Imputation

* **KNN Imputation:** Missing `Speed_of_Impact` values are estimated using a K-Nearest Neighbors approach based on `Age`.


* **Mode Imputation:** Missing `Gender` entries are filled using the most frequent value.


* **Pipeline Integration:** Numeric features are scaled via `StandardScaler`, and categorical features are processed using `OneHotEncoder`.



### ### 2. Feature Engineering

New features were derived to capture complex interactions:

* **Binned Categories:** `Age` and `Speed_of_Impact` are transformed into categorical bins (e.g., "Young Adult", "Extreme Speed").


* **Interaction Terms:** Features like `Age_Speed_Interaction` and `Helmet_Survived` capture combined effects of different variables.


* **Polynomial Features:** `Age_Squared` and `Speed_Squared` allow the model to capture non-linear relationships.



### ### 3. Handling Class Imbalance

* **SMOTE:** Synthetic Minority Over-sampling Technique is applied to the training data to ensure the models are not biased toward the majority class.



---

## ## Modeling & Evaluation

### ### Machine Learning Models

The pipeline evaluates multiple high-performance algorithms:

* **Random Forest:** Utilizes 1000 estimators with balanced class weights.


* **Gradient Boosting & XGBoost:** Gradient-boosted decision trees optimized for sequential learning.


* **Stacking Classifier:** An ensemble that combines RF, GB, and XGBoost predictions using `LogisticRegression` as a final meta-learner.



### ### Performance Tuning

* **GridSearchCV:** Automated hyperparameter tuning (n_estimators, max_depth, etc.) using `StratifiedKFold` cross-validation to maximize accuracy.



### ### Evaluation Metrics

* **Confusion Matrix:** Visualizes true/false positives and negatives.


* **ROC/AUC:** Measures the model's ability to distinguish between classes.


* **Feature Importance:** Identifies the most significant predictors (e.g., `Age`, `Speed_of_Impact`, `Helmet_Used`).



---

## ## Visualizations Generated

The script produces several analytical plots:

* `seatbelt_distribution.png`: Initial class balance view.


* `confusion_matrix_tuned.png`: Predictive performance of the final model.


* `roc_curve.png` & `precision_recall_curve.png`: Model reliability metrics.


* `feature_importance_tuned.png`: Visual ranking of variables influencing seatbelt usage.



---

## ## Summary of Findings

* **Model Accuracy:** The pipeline provides both baseline and tuned performance scores to track improvements.


* **What-if Analysis:** The model allows for individual instance analysis, such as predicting how changing a "Helmet Status" impacts the probability of "Seatbelt Usage."



---

## ## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```[cite: 1]

```
