import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import warnings
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
warnings.filterwarnings('ignore')

# Load the dataset
file_path = './accident.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nMissing values before handling:")
print(data.isnull().sum())

# 1. IMPROVED DATA PREPROCESSING

# More advanced missing value imputation
# Use KNN imputation for Speed_of_Impact
knn_imputer = KNNImputer(n_neighbors=5)
data['Speed_of_Impact'] = knn_imputer.fit_transform(data[['Speed_of_Impact', 'Age']])[:, 0]

# Handle missing values in Gender using mode
if data['Gender'].isnull().sum() > 0:
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

# 2. FEATURE ENGINEERING
# Creating new features that might help improve prediction

# Age categories
data['Age_Category'] = pd.cut(data['Age'], 
                             bins=[0, 20, 30, 40, 50, 60, 70, 100], 
                             labels=['Very_Young', 'Young', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior', 'Elderly'])

# Speed categories
data['Speed_Category'] = pd.cut(data['Speed_of_Impact'], 
                               bins=[0, 20, 40, 60, 80, 100, 120], 
                               labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High', 'Extreme'])

# Interaction features
data['Helmet_Survived'] = data['Helmet_Used'] + '_' + data['Survived'].astype(str)

# Convert categorical target to binary
data['Seatbelt_Used'] = data['Seatbelt_Used'].map({'Yes': 1, 'No': 0})

# Add more interaction features and polynomial features
data['Age_Speed_Interaction'] = data['Age'] * data['Speed_of_Impact']
data['Age_Squared'] = data['Age'] ** 2
data['Speed_Squared'] = data['Speed_of_Impact'] ** 2

# Time of day features (if available in your dataset)
if 'Time' in data.columns:
    data['Hour'] = pd.to_datetime(data['Time']).dt.hour
    data['Is_Night'] = (data['Hour'] >= 22) | (data['Hour'] <= 5)

# Weather conditions (if available)
if 'Weather' in data.columns:
    data['Is_Bad_Weather'] = data['Weather'].isin(['Rain', 'Snow', 'Fog'])

print("\nData after feature engineering:")
print(data.head())

# 3. VISUALIZE CLASS DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.countplot(x='Seatbelt_Used', data=data)
plt.title('Seatbelt Usage Distribution')
plt.savefig('seatbelt_distribution.png')
plt.close()

# Check class imbalance
seatbelt_counts = data['Seatbelt_Used'].value_counts()
print("\nSeatbelt usage distribution:")
print(seatbelt_counts)
print(f"Proportion of seatbelt usage: {seatbelt_counts[1]/len(data):.4f}")

# 4. PREPARE DATA FOR MODELING
# Split features and target
X = data.drop(['Seatbelt_Used'], axis=1)
y = data['Seatbelt_Used']

# Create train-test split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Identify categorical and numerical columns
categorical_cols = ['Gender', 'Helmet_Used', 'Survived', 'Age_Category', 'Speed_Category', 'Helmet_Survived']
numerical_cols = ['Age', 'Speed_of_Impact']

# Advanced preprocessing with imputation strategies
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 5. HANDLE CLASS IMBALANCE USING SMOTE
# Apply SMOTE to training data only
smote = SMOTE(random_state=42)

# First preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"\nClass distribution before SMOTE: {np.bincount(y_train)}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

# 6. ADVANCED MODEL SELECTION
# Define models with better default parameters
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=2,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# Dictionary to store results
model_results = {}

# Train and evaluate each model
print("\nBaseline model performance:")
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_results[name] = acc
    print(f"{name} accuracy: {acc:.4f}")

# Identify best model
best_model_name = max(model_results, key=model_results.get)
print(f"\nBest performing model: {best_model_name} with accuracy {model_results[best_model_name]:.4f}")

# 7. TRAIN ENSEMBLE MODEL
# Create a stacking classifier with the best models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=500, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=500, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=500, random_state=42))
]

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Create pipeline with preprocessor and stacking classifier
ensemble_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_classifier)
])

# Train ensemble model
ensemble_pipeline.fit(X_train, y_train)

# 8. HYPERPARAMETER TUNING
# Simplified parameter grid for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [500, 1000],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced']
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [500, 1000],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }
else:  # XGBoost
    param_grid = {
        'classifier__n_estimators': [500, 1000],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }

# Create pipeline with best model
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Perform grid search
print(f"\nTuning hyperparameters for {best_model_name}...")
grid_search = GridSearchCV(
    best_pipeline,
    param_grid=param_grid,
    cv=StratifiedKFold(5),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get best model
tuned_model = grid_search.best_estimator_

# 9. EVALUATE TUNED MODEL
# Make predictions
y_pred_tuned = tuned_model.predict(X_test)

# Evaluate
tuned_acc = accuracy_score(y_test, y_pred_tuned)
print(f"\nTuned model accuracy: {tuned_acc:.4f}")
print("\nTuned model classification report:")
print(classification_report(y_test, y_pred_tuned))

# Confusion matrix for tuned model
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Seatbelt', 'Seatbelt Used'],
            yticklabels=['No Seatbelt', 'Seatbelt Used'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Tuned Model)')
plt.savefig('confusion_matrix_tuned.png')
plt.close()

# 10. ROC CURVE AND AUC
# Get prediction probabilities
y_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_proba_tuned)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# 11. PRECISION-RECALL CURVE
# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba_tuned)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig('precision_recall_curve.png')
plt.close()

# 12. FEATURE IMPORTANCE ANALYSIS
# Get feature names from preprocessor
feature_names = tuned_model.named_steps['preprocessor'].get_feature_names_out()

# Get feature importance if the model supports it
if hasattr(tuned_model.named_steps['classifier'], 'feature_importances_'):
    feature_importances = tuned_model.named_steps['classifier'].feature_importances_
    
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_tuned.png')
    plt.close()
    
    # Print top 10 important features
    print("\nTop 10 important features:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}")
else:
    # Use permutation importance as an alternative
    perm_importance = permutation_importance(tuned_model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importances')
    plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], color='b', align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    plt.close()
    
    # Print top 10 important features
    print("\nTop 10 important features (permutation importance):")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[sorted_idx[i]]}: {perm_importance.importances_mean[sorted_idx[i]]:.4f}")

# 13. ANALYZE SELECTED INSTANCE
# Instance #42 (24-year-old male with helmet but no seatbelt)
selected_instance = X.iloc[42-1:42].copy()
probability_tuned = tuned_model.predict_proba(selected_instance)[0]

print("\nPrediction for Selected Instance (Tuned Model):")
print(f"Probability of seatbelt usage: {probability_tuned[1]:.4f}")
print(f"Actual outcome: {'Seatbelt Used' if data.iloc[42-1]['Seatbelt_Used'] == 1 else 'No Seatbelt'}")

# What-if analysis - changing helmet status
selected_instance_modified = selected_instance.copy()
original_helmet = selected_instance['Helmet_Used'].iloc[0]
selected_instance_modified['Helmet_Used'] = 'No' if original_helmet == 'Yes' else 'Yes'

# Update the derived feature too
selected_instance_modified['Helmet_Survived'] = selected_instance_modified['Helmet_Used'] + '_' + selected_instance_modified['Survived'].astype(str)

probability_modified_tuned = tuned_model.predict_proba(selected_instance_modified)[0]

print("\nWhat-if Analysis (changing helmet status):")
print(f"Original helmet status: {original_helmet}")
print(f"Original probability of seatbelt usage: {probability_tuned[1]:.4f}")
print(f"Modified helmet status: {'No' if original_helmet == 'Yes' else 'Yes'}")
print(f"Modified probability of seatbelt usage: {probability_modified_tuned[1]:.4f}")
print(f"Change in seatbelt usage probability: {probability_modified_tuned[1] - probability_tuned[1]:.4f}")

# 14. SUMMARY AND RECOMMENDATIONS
print("\n============ SUMMARY ============")
print(f"1. Initial model accuracy: {model_results[best_model_name]:.4f}")
print(f"2. Ensemble model accuracy: {tuned_acc:.4f}")
print("3. Key improvement techniques used:")
print("   - Advanced feature engineering")
print("   - Handling class imbalance with SMOTE")
print("   - Ensemble learning")
print("   - Hyperparameter tuning")
print("   - Advanced imputation techniques")
print("4. Next steps for further improvement:")
print("   - Collect more data")
print("   - Try different feature combinations")
print("   - Implement more advanced ensemble techniques")
print("   - Consider deep learning approaches for complex patterns")
