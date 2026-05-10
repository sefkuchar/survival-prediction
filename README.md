# Traffic Safety Compliance and Accident Analysis Framework

This repository provides a professional, end-to-end machine learning solution for analyzing occupant safety behavior. The system utilizes advanced ensemble modeling and feature engineering to predict safety equipment usage (seatbelts) and identify high-risk demographics in vehicular accidents.

---

## Analytical Objectives

The framework is designed to address two primary research and operational tasks:

1.  **Predictive Compliance Modeling:** Building a high-precision classifier to determine the probability of seatbelt usage based on situational factors (Speed, Age, Gender) and secondary safety equipment (Helmets).
2.  **Safety Correlation Analysis:** Identifying specific occupant segments where safety compliance is significantly lower than average to inform targeted public safety campaigns and legislative interventions.

---

## Technical Workflow

### 1. Advanced Data Engineering
*   **Intelligent Imputation:** Implementation of `KNNImputer` for `Speed_of_Impact` to maintain data variance by leveraging relationships between age and impact velocity.
*   **Non-Linear Feature Synthesis:** Generation of polynomial terms ($Age^2$, $Speed^2$) and interaction variables ($Age \times Speed$) to capture complex, non-linear correlations.
*   **Synthetic Resampling:** Application of **SMOTE** to the training pipeline to balance the target classes, ensuring the model remains sensitive to the minority class (non-compliance).

### 2. Modeling Architecture
The system employs a multi-stage **Stacking Ensemble** approach:
*   **Base Layer:** Independent training of `Random Forest`, `Gradient Boosting`, and `XGBoost` classifiers.
*   **Meta Layer:** A `Logistic Regression` final estimator that synthesizes the predictions of the base layer to reduce variance and improve generalization.
*   **Optimization:** Automated hyperparameter tuning via `GridSearchCV` with `StratifiedKFold` cross-validation to maximize the F1-score and Accuracy.

---

## Key Performance Metrics

The framework evaluates the "Safety Compliance Model" through three critical lenses:

*   **Reliability (Confidence):** Measured via the Precision-Recall curve to ensure high-probability predictions are statistically sound.
*   **Impact (Feature Importance):** Using Gini and Permutation Importance to rank factors; identifying that variables like `Helmet_Used` and `Age_Category` are primary drivers of seatbelt usage.
*   **Discrimination (AUC-ROC):** Evaluating the model's ability to distinguish between compliant and non-compliant occupants across varying thresholds.

---

## Strategic Insights Generated

### Compliance Drivers
*   **Segment Identification:** The model identifies specific "Risk Segments" (e.g., young occupants in high-speed impacts) where compliance drops significantly.
*   **Behavioral Links:** Quantitative validation of the link between different safety behaviors (e.g., the likelihood of seatbelt use among those already utilizing helmets).

### What-If Simulations
*   The script includes a simulation module to test how changing a single variable (e.g., providing a helmet) shifts the predicted probability of seatbelt compliance for a specific demographic.

---

## Visualization Suite
The pipeline automates the generation of professional-grade diagnostic plots:
*   `confusion_matrix_tuned.png`: Detailed breakdown of classification accuracy.
*   `roc_curve.png`: Visual representation of the True Positive vs. False Positive rate.
*   `feature_importance_tuned.png`: Hierarchical ranking of the most influential predictors in the dataset.

---

## Deployment Requirements
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
