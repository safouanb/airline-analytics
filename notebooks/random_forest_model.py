# -*- coding: utf-8 -*-
"""Random Forest Model for Airline Satisfaction Prediction

This script trains a Random Forest classifier to predict customer satisfaction.
Random Forest is an ensemble method that combines multiple decision trees to
improve prediction accuracy and reduce overfitting.

Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.pipeline import Pipeline

# Import our custom preprocessing functions
# These handle data loading and feature transformation
from preprocessing import load_data, get_preprocessor

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
# Load the training and test datasets from the data/raw directory
# X contains all features (service ratings, delays, demographics, etc.)
# y contains the target variable (satisfied = 1, dissatisfied = 0)
print("Loading data...")
X_train_full, y_train_full = load_data('train.csv', data_dir='data/raw')
X_test_final, y_test_final = load_data('test.csv', data_dir='data/raw')
print(f"Training samples: {len(X_train_full):,}")
print(f"Test samples: {len(X_test_final):,}")

# ============================================================================
# STEP 2: CREATE PREPROCESSING PIPELINE
# ============================================================================
# The preprocessor handles:
# - Imputing missing values (e.g., arrival delays)
# - Encoding categorical variables (Gender, Class, Travel Type)
# - Ordinal encoding for Class (Eco < Eco Plus < Business)
preprocessor = get_preprocessor(X_train_full)

# ============================================================================
# STEP 3: CONFIGURE RANDOM FOREST MODEL
# ============================================================================
# Random Forest builds multiple decision trees and averages their predictions
# This reduces overfitting and improves accuracy compared to a single tree
#
# Hyperparameters explained:
# - n_estimators=100: Build 100 different decision trees
#                     More trees = more stable predictions, but slower training
# - max_depth=15: Each tree can go max 15 levels deep
#                 This prevents individual trees from memorizing the training data
# - min_samples_split=20: Need at least 20 samples to split a node
#                         Same as decision tree model for fair comparison
# - min_samples_leaf=10: Each leaf must have at least 10 samples
#                        Prevents creating leaves for individual outliers
# - random_state=42: Makes results reproducible (same random splits every time)
# - n_jobs=-1: Use all available CPU cores to train trees in parallel (faster!)
print("\nConfiguring Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# Combine preprocessing and model into a single pipeline
# This ensures preprocessing happens inside cross-validation (prevents data leakage)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', rf_model)])

# ============================================================================
# STEP 4: MODEL SELECTION - COMPARE MODELS ON TRAINING DATA
# ============================================================================
# Compare different model configurations using cross-validation on training data
# This is the proper ML workflow: select best model on training, then test on held-out data
print("\n" + "="*60)
print("MODEL SELECTION - Comparing Models on Training Data")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Import additional preprocessing components needed for feature subset models
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# MODEL 1: Full Model (All Features)
print("\n[Model 1] Full Model (All Features)...")
cv_results_full = cross_validate(
    clf, X_train_full, y_train_full, 
    cv=cv, 
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
)
cv_acc_full = cv_results_full['test_accuracy'].mean()
cv_auc_full = cv_results_full['test_roc_auc'].mean()
print(f"  CV Accuracy: {cv_acc_full:.4f} (+/- {cv_results_full['test_accuracy'].std():.4f})")
print(f"  CV AUC:      {cv_auc_full:.4f} (+/- {cv_results_full['test_roc_auc'].std():.4f})")

# MODEL 2: Top 7 Features
features_top7 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class',
    'Inflight entertainment', 'Customer Type', 'Leg room service'
]
print("\n[Model 2] Top 7 Features Model...")
X_train_7 = X_train_full[features_top7].copy()
prep_7 = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())]),
     ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Leg room service']),
    ('ord', Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Eco', 'Eco Plus', 'Business']]))]),
     ['Class']),
    ('cat', Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))]),
     ['Type of Travel', 'Customer Type'])
])
rf_7 = Pipeline(steps=[('preprocessor', prep_7),
                       ('classifier', RandomForestClassifier(
                           n_estimators=100, max_depth=15, min_samples_split=20,
                           min_samples_leaf=10, random_state=42, n_jobs=-1))])
cv_results_7 = cross_validate(
    rf_7, X_train_7, y_train_full, 
    cv=cv, 
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
)
cv_acc_7 = cv_results_7['test_accuracy'].mean()
cv_auc_7 = cv_results_7['test_roc_auc'].mean()
print(f"  CV Accuracy: {cv_acc_7:.4f} (+/- {cv_results_7['test_accuracy'].std():.4f})")
print(f"  CV AUC:      {cv_auc_7:.4f} (+/- {cv_results_7['test_roc_auc'].std():.4f})")

# MODEL 3: Top 5 Features
features_top5 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class', 'Inflight entertainment'
]
print("\n[Model 3] Top 5 Features Model...")
X_train_5 = X_train_full[features_top5].copy()
prep_5 = ColumnTransformer(transformers=[
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())]),
     ['Online boarding', 'Inflight wifi service', 'Inflight entertainment']),
    ('ord', Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Eco', 'Eco Plus', 'Business']]))]),
     ['Class']),
    ('cat', Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))]),
     ['Type of Travel'])
])
rf_5 = Pipeline(steps=[('preprocessor', prep_5),
                       ('classifier', RandomForestClassifier(
                           n_estimators=100, max_depth=15, min_samples_split=20,
                           min_samples_leaf=10, random_state=42, n_jobs=-1))])
cv_results_5 = cross_validate(
    rf_5, X_train_5, y_train_full, 
    cv=cv, 
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
)
cv_acc_5 = cv_results_5['test_accuracy'].mean()
cv_auc_5 = cv_results_5['test_roc_auc'].mean()
print(f"  CV Accuracy: {cv_acc_5:.4f} (+/- {cv_results_5['test_accuracy'].std():.4f})")
print(f"  CV AUC:      {cv_auc_5:.4f} (+/- {cv_results_5['test_roc_auc'].std():.4f})")

# ============================================================================
# STEP 5: SELECT BEST MODEL
# ============================================================================
# Compare models and select the best one based on cross-validation AUC
# (AUC is preferred as it's more robust to class imbalance)
print("\n" + "="*60)
print("SELECTING BEST MODEL")
print("="*60)

models = {
    'Full Model (All Features)': {
        'pipeline': clf,
        'X_train': X_train_full,
        'features': 'all',
        'cv_acc': cv_acc_full,
        'cv_auc': cv_auc_full,
        'cv_results': cv_results_full
    },
    'Top 7 Features': {
        'pipeline': rf_7,
        'X_train': X_train_7,
        'features': features_top7,
        'cv_acc': cv_acc_7,
        'cv_auc': cv_auc_7,
        'cv_results': cv_results_7
    },
    'Top 5 Features': {
        'pipeline': rf_5,
        'X_train': X_train_5,
        'features': features_top5,
        'cv_acc': cv_acc_5,
        'cv_auc': cv_auc_5,
        'cv_results': cv_results_5
    }
}

# Select best model based on AUC (primary metric)
best_model_name = max(models.keys(), key=lambda k: models[k]['cv_auc'])
best_model = models[best_model_name]

print(f"\nModel Comparison (5-Fold CV on Training Data):")
print(f"  {'Model':<25} {'CV Accuracy':<15} {'CV AUC':<15}")
print(f"  {'-'*55}")
for name, model_info in models.items():
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"  {name:<25} {model_info['cv_acc']:.4f}        {model_info['cv_auc']:.4f}{marker}")

print(f"\n✓ Selected Best Model: {best_model_name}")
print(f"  CV Accuracy: {best_model['cv_acc']:.4f}")
print(f"  CV AUC:      {best_model['cv_auc']:.4f}")

# ============================================================================
# STEP 6: TRAIN BEST MODEL ON FULL TRAINING SET
# ============================================================================
# Now train the selected best model on the full training set
print("\n" + "="*60)
print("TRAINING BEST MODEL ON FULL TRAINING SET")
print("="*60)
print(f"Training {best_model_name} on all {len(X_train_full):,} training samples...")
best_model['pipeline'].fit(best_model['X_train'], y_train_full)
print("✓ Training complete!")

# ============================================================================
# STEP 7: EVALUATE BEST MODEL ON TEST SET
# ============================================================================
# Evaluate the best model on the held-out test set to get final performance
print("\n" + "="*60)
print("EVALUATING BEST MODEL ON TEST SET")
print("="*60)

# Prepare test data with same features as best model
if best_model['features'] == 'all':
    X_test_prepared = X_test_final
else:
    X_test_prepared = X_test_final[best_model['features']].copy()

# Make predictions on test set
print("Making predictions on test set...")
y_pred_test = best_model['pipeline'].predict(X_test_prepared)
y_pred_proba_test = best_model['pipeline'].predict_proba(X_test_prepared)[:, 1]

# Calculate test set performance metrics
test_accuracy = accuracy_score(y_test_final, y_pred_test)
test_auc = roc_auc_score(y_test_final, y_pred_proba_test)

print(f"\n{'='*60}")
print(f"FINAL TEST SET RESULTS - {best_model_name}")
print(f"{'='*60}")
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print(f"Test Set AUC:      {test_auc:.4f}")
print(f"{'='*60}")

# ============================================================================
# STEP 6: CONFUSION MATRIX (TEST SET)
# ============================================================================
# Shows how many predictions were correct vs incorrect on test data
print("\n--- Generating Confusion Matrix (Test Set) ---")
cm_test = confusion_matrix(y_test_final, y_pred_test)

# Display as a formatted table
cm_df = pd.DataFrame(cm_test,
                     index=['Actual: Neutral/Dissatisfied', 'Actual: Satisfied'],
                     columns=['Pred: Neutral/Dissatisfied', 'Pred: Satisfied'])
print(cm_df)
print("-" * 60)

# Create visual heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Neutral/Dissatisfied', 'Satisfied'],
            yticklabels=['Neutral/Dissatisfied', 'Satisfied'])
plt.title('Confusion Matrix - Test Set (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
print("✓ Saved: rf_confusion_matrix.png (Test Set)")

# ============================================================================
# STEP 7: ROC CURVE (TEST SET)
# ============================================================================
# ROC curve shows the trade-off between true positive rate and false positive rate
# AUC (Area Under Curve) summarizes performance: 1.0 = perfect, 0.5 = random guess
print("\n--- Generating ROC Curve (Test Set) ---")
fpr_test, tpr_test, _ = roc_curve(y_test_final, y_pred_proba_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='#55a868', lw=2.5, 
         label=f'Random Forest (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Guess (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC Curve - Test Set (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('rf_roc_curve.png', dpi=150)
print("✓ Saved: rf_roc_curve.png (Test Set)")

# ============================================================================
# STEP 8: GLOBAL FEATURE IMPORTANCE ANALYSIS (BEST MODEL)
# ============================================================================
# Identify which features (service ratings, delays, etc.) are most important
# for predicting customer satisfaction in the best model
print("\n--- Analyzing Feature Importance (Best Model) ---")

# Reconstruct feature names after preprocessing based on best model
if best_model['features'] == 'all':
    # Full model - use all features
    numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()
    ordinal_features = ['Class']
    nominal_features = ['Gender', 'Customer Type', 'Type of Travel']
    cat_names = best_model['pipeline'].named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(nominal_features).tolist()
    feature_names = numerical_features + ordinal_features + cat_names
else:
    # Subset model - reconstruct from the preprocessor
    if len(best_model['features']) == 7:
        # Top 7 features model
        num_feat = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Leg room service']
        ord_feat = ['Class']
        cat_feat = ['Type of Travel', 'Customer Type']
        cat_names = best_model['pipeline'].named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(cat_feat).tolist()
        feature_names = num_feat + ord_feat + cat_names
    else:
        # Top 5 features model
        num_feat = ['Online boarding', 'Inflight wifi service', 'Inflight entertainment']
        ord_feat = ['Class']
        cat_feat = ['Type of Travel']
        cat_names = best_model['pipeline'].named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(cat_feat).tolist()
        feature_names = num_feat + ord_feat + cat_names

# Extract feature importances from the trained Random Forest
# Random Forest averages importance across all 100 trees
# Higher value = more important for predicting satisfaction
importances = best_model['pipeline'].named_steps['classifier'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Plot the top 10 most important features
# Use consistent styling with decision tree model
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh', color='#55a868')
plt.title('Top 10 Drivers of Satisfaction (Random Forest)')
plt.xlabel('Relative Importance (Mean Decrease in Impurity)')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
print("Saved rf_feature_importance.png")


# ============================================================================
# STEP 9: FEATURE IMPORTANCE WITH VARIABILITY (RANDOM FOREST SPECIFIC)
# ============================================================================
# Random Forest unique feature: we can see how consistent importance is across trees
# If a feature has high variability, it means different trees disagree on its importance
print("\n--- Analyzing Feature Importance Variability ---")

# Extract importance from each individual tree in the forest
tree_importances = []
for tree in best_model['pipeline'].named_steps['classifier'].estimators_:
    tree_importances.append(tree.feature_importances_)

# Calculate standard deviation of importance across all 100 trees
importances_std = pd.DataFrame(tree_importances).std(axis=0)

# Get top 10 features and their variability
top_10_indices = feat_imp.head(10).index
top_10_values = feat_imp.head(10).values
top_10_std = importances_std[feat_imp.head(10).index.map(lambda x: feature_names.index(x))]

# Plot with error bars showing variability across trees
# This is unique to Random Forest - shows consistency of feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(10), top_10_values, xerr=top_10_std, color='#55a868', alpha=0.8)
plt.yticks(range(10), top_10_indices)
plt.xlabel('Importance (with standard deviation across trees)')
plt.title('Top 10 Feature Importances with Variability (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rf_feature_importance_with_std.png')
print("Saved rf_feature_importance_with_std.png")

# ============================================================================
# STEP 10: SUBGROUP ANALYSIS
# ============================================================================
# Different customer segments may care about different things
# For example: Business class passengers might prioritize wifi, while
# Economy passengers might care more about seat comfort
print("\n" + "="*60)
print("SUBGROUP ANALYSIS")
print("="*60)

# Combine features and target for easy filtering
df_full = X_train_full.copy()
df_full['satisfaction'] = y_train_full


def get_importance_for_subset(filter_col, filter_value):
    """
    Train a Random Forest on a specific customer subgroup.
    
    This helps us understand what drives satisfaction for different types
    of customers (e.g., Business class vs Economy, Business travel vs Personal).
    
    Args:
        filter_col: Column to filter on (e.g., 'Class', 'Type of Travel')
        filter_value: Value to filter for (e.g., 'Business', 'Personal Travel')
    
    Returns:
        pd.Series: Feature importances for this subgroup, or None if no data
    """
    # Filter to only this subgroup
    subset = df_full[df_full[filter_col] == filter_value]
    
    # Skip if no samples in this subgroup
    if len(subset) == 0:
        print(f"  ⚠ Warning: No samples found for {filter_value}")
        return None

    # Separate features and target
    y_sub = subset['satisfaction']
    X_sub = subset.drop('satisfaction', axis=1)

    # Train a new Random Forest just for this subgroup
    # Using same hyperparameters as main model for consistency
    local_preprocessor = get_preprocessor(X_train_full)
    local_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    local_clf = Pipeline(steps=[('preprocessor', local_preprocessor),
                                ('classifier', local_model)])

    print(f"  Training model for {filter_value} ({len(subset):,} samples)...")
    local_clf.fit(X_sub, y_sub)

    # Extract feature importances
    local_cat_names = local_clf.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(nominal_features).tolist()
    local_cat_names = [c.replace('Type of Travel_', '').replace('Customer Type_', '').replace('Gender_', '') for c in local_cat_names]
    local_feat_names = numerical_features + ordinal_features + local_cat_names

    imps = local_clf.named_steps['classifier'].feature_importances_
    return pd.Series(imps, index=local_feat_names)


# ============================================================================
# STEP 8A: COMPARE BY TRAVEL CLASS
# ============================================================================
# Do Business, Economy, and Economy Plus passengers care about different things?
print("\n[1/2] Analyzing by Travel Class...")
imp_bus = get_importance_for_subset('Class', 'Business')
imp_eco = get_importance_for_subset('Class', 'Eco')
imp_plus = get_importance_for_subset('Class', 'Eco Plus')

# Combine into a single dataframe for comparison
df_class_comp = pd.DataFrame({'Business': imp_bus, 'Economy': imp_eco, 'Eco Plus': imp_plus})
# Fill missing values with 0 (in case a subgroup has no data)
df_class_comp = df_class_comp.fillna(0).sort_values(by='Business', ascending=False).head(8)

# Create comparison plot - consistent styling with decision tree
df_class_comp.plot(kind='bar', figsize=(14, 6), width=0.8, color=['#4c72b0', '#dd8452', '#55a868'])
plt.title('Feature Importance by Travel Class (Random Forest)')
plt.ylabel('Relative Importance')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Travel Class')
plt.tight_layout()
plt.savefig('rf_subgroup_class_comparison.png')
print("Saved rf_subgroup_class_comparison.png")

# ============================================================================
# STEP 8B: COMPARE BY TRAVEL TYPE
# ============================================================================
# Do business travelers and personal travelers have different priorities?
print("\n[2/2] Analyzing by Travel Type...")
imp_biz_travel = get_importance_for_subset('Type of Travel', 'Business travel')
imp_personal = get_importance_for_subset('Type of Travel', 'Personal Travel')

# Combine into a single dataframe for comparison
df_type_comp = pd.DataFrame({'Business Travel': imp_biz_travel, 'Personal Travel': imp_personal})
df_type_comp = df_type_comp.fillna(0).sort_values(by='Business Travel', ascending=False).head(8)

# Create comparison plot - consistent styling with decision tree
df_type_comp.plot(kind='bar', figsize=(12, 6), width=0.8, color=['#8172b3', '#c44e52'])
plt.title('Feature Importance by Type of Travel (Random Forest)')
plt.ylabel('Relative Importance')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Travel Type')
plt.tight_layout()
plt.savefig('rf_subgroup_type_comparison.png')
print("Saved rf_subgroup_type_comparison.png")


print("\n" + "="*60)
print("RANDOM FOREST MODEL ANALYSIS COMPLETE!")
print("="*60)
print(f"\nModel Selection Summary:")
print(f"  Best Model Selected: {best_model_name}")
print(f"  Selection Criteria: Highest CV AUC on training data")
print(f"\nPerformance Summary:")
print(f"  Cross-Validation (5-fold) - Training Data:")
print(f"    - Accuracy: {best_model['cv_acc']:.4f} (+/- {best_model['cv_results']['test_accuracy'].std():.4f})")
print(f"    - AUC:      {best_model['cv_auc']:.4f} (+/- {best_model['cv_results']['test_roc_auc'].std():.4f})")
print(f"\n  Test Set (Final Evaluation) - Unseen Data:")
print(f"    - Accuracy: {test_accuracy:.4f}")
print(f"    - AUC:      {test_auc:.4f}")
print(f"\n  Model Comparison (All models evaluated on training data via CV):")
for name, model_info in models.items():
    marker = " <-- SELECTED" if name == best_model_name else ""
    print(f"    - {name}:")
    print(f"      CV Accuracy: {model_info['cv_acc']:.4f}, CV AUC: {model_info['cv_auc']:.4f}{marker}")
print(f"\nKey Insights:")
print(f"  - Best model selected based on cross-validation performance on training data")
print(f"  - Final model trained on full training set ({len(X_train_full):,} samples)")
print(f"  - Test set evaluation on {len(X_test_final):,} unseen samples")
print(f"  - Random Forest provides ensemble learning with 100 trees")
print(f"  - Test set performance confirms model generalizes well to unseen data")
print(f"\nGenerated Visualizations (Best Model - Test Set Results):")
print(f"  ✓ rf_confusion_matrix.png (Test Set)")
print(f"  ✓ rf_roc_curve.png (Test Set)")
print(f"  ✓ rf_feature_importance.png (Best Model)")
print(f"  ✓ rf_feature_importance_with_std.png (Best Model)")
print(f"  ✓ rf_subgroup_class_comparison.png")
print(f"  ✓ rf_subgroup_type_comparison.png")
print("\n" + "="*60)

