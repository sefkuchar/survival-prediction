## Project Overview
The primary goal is to analyze accident data to identify factors influencing seatbelt usage and build a robust classifier. The project addresses common real-world data challenges such as missing values and class imbalance.

---

## Technical Workflow

### 1. Advanced Preprocessing & Imputation
* **KNN Imputation:** Missing `Speed_of_Impact` values are estimated using a K-Nearest Neighbors approach based on `Age`.
* **Mode Imputation:** Missing `Gender` entries are filled using the most frequent value.
* **Pipeline Integration:** Numeric features are scaled via `StandardScaler`, and categorical features are processed using `OneHotEncoder` within a `ColumnTransformer`.

### 2. Feature Engineering
New features were derived to capture complex interactions:
* **Binned Categories:** `Age` and `Speed_of_Impact` are transformed into categorical bins (e.g., "Young Adult", "Extreme Speed").
* **Interaction Terms:** Features like `Age_Speed_Interaction` and `Helmet_Survived` capture combined effects.
* **Polynomial Features:** `Age_Squared` and `Speed_Squared` allow the model to capture non-linear relationships.

### 3. Handling Class Imbalance
* **SMOTE:** Synthetic Minority Over-sampling Technique is applied to the training data to ensure the models are not biased toward the majority class (No Seatbelt vs. Seatbelt Used).

---

## Modeling & Evaluation

### Machine Learning Models
The pipeline evaluates and compares several high-performance algorithms:
* **Random Forest:** 1000 estimators with balanced class weights.
* **Gradient Boosting & XGBoost:** Gradient-boosted decision trees optimized for sequential learning.
* **Stacking Classifier:** An ensemble that combines RF, GB, and XGBoost predictions using a Logistic Regression meta-learner.

### Performance Tuning
* **GridSearchCV:** Automated hyperparameter tuning using `StratifiedKFold` cross-validation to maximize accuracy and ensure generalizability.

### Evaluation Metrics
The script generates a full suite of metrics:
* **Confusion Matrix:** To visualize classification errors.
* **ROC/AUC & Precision-Recall:** To evaluate model performance across different thresholds.
* **Feature Importance:** Using both native model importance and **Permutation Importance** to identify key safety predictors.

---

## Visualizations Generated
The script produces several analytical plots:
* `seatbelt_distribution.png`: Initial class balance view.
* `confusion_matrix_tuned.png`: Predictive performance of the final model.
* `roc_curve.png`: Model sensitivity and specificity.
* `feature_importance_tuned.png`: Visual ranking of variables influencing seatbelt usage.

---

## Summary of Findings
* **Individual Instance Analysis:** Includes a "What-if" scenario to see how changing one variable (like Helmet use) affects the predicted probability of seatbelt use.
* **Key Techniques:** Demonstrates the use of pipelines, SMOTE, stacking, and advanced imputation in a single workflow.

---

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
