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
# Load the training dataset from the data/raw directory
# Following same approach as decision tree model - focus on cross-validation
# X contains all features (service ratings, delays, demographics, etc.)
# y contains the target variable (satisfied = 1, dissatisfied = 0)
print("Loading training data...")
X_train_full, y_train_full = load_data('train.csv', data_dir='data/raw')
print(f"Training samples: {len(X_train_full):,}")

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
# STEP 4: CROSS-VALIDATION
# ============================================================================
# Use 5-fold cross-validation to estimate model performance
# Stratified k-fold maintains the same ratio of satisfied/dissatisfied in each fold
# This gives us a more reliable estimate of how the model will perform on new data
print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate the model 5 times (once per fold)
# We compute both accuracy and AUC in a single pass for efficiency
cv_results = cross_validate(
    clf, X_train_full, y_train_full, 
    cv=cv, 
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
)

# Display cross-validation results
# Focus on cross-validation performance instead of test set evaluation
# This approach matches the decision tree model and follows best practices
print(f"\n--- FULL MODEL 5-Fold Cross-Validation Results ---")
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Mean AUC: {cv_results['test_roc_auc'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 5: TRAIN FINAL MODEL FOR FEATURE ANALYSIS
# ============================================================================
# Train the model on full training set to analyze feature importance
# We focus on cross-validation results for performance evaluation
print("\nTraining full model for feature importance analysis...")
clf.fit(X_train_full, y_train_full)

# ============================================================================
# STEP 6: GLOBAL FEATURE IMPORTANCE ANALYSIS
# ============================================================================
# Identify which features (service ratings, delays, etc.) are most important
# for predicting customer satisfaction
print("\n--- Analyzing Feature Importance ---")

# Reconstruct feature names after preprocessing
# Numerical features stay the same (Age, Flight Distance, delays, service ratings)
numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()
# Ordinal feature (Class) gets encoded as a single number
ordinal_features = ['Class']
# Nominal features get one-hot encoded (Gender_Male, Type of Travel_Business, etc.)
nominal_features = ['Gender', 'Customer Type', 'Type of Travel']
cat_names = clf.named_steps['preprocessor'].named_transformers_['cat']['encoder'].get_feature_names_out(nominal_features).tolist()

# Combine all feature names in the same order as the model sees them
feature_names = numerical_features + ordinal_features + cat_names

# Extract feature importances from the trained Random Forest
# Random Forest averages importance across all 100 trees
# Higher value = more important for predicting satisfaction
importances = clf.named_steps['classifier'].feature_importances_
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
# STEP 7: FEATURE IMPORTANCE WITH VARIABILITY (RANDOM FOREST SPECIFIC)
# ============================================================================
# Random Forest unique feature: we can see how consistent importance is across trees
# If a feature has high variability, it means different trees disagree on its importance
print("\n--- Analyzing Feature Importance Variability ---")

# Extract importance from each individual tree in the forest
tree_importances = []
for tree in clf.named_steps['classifier'].estimators_:
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
# STEP 8: SUBGROUP ANALYSIS
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


# ============================================================================
# STEP 9: MULTI-MODEL COMPARISON (INSPIRED BY DECISION TREE ANALYSIS)
# ============================================================================
# Following the approach from the decision tree model, we'll compare:
# 1. Full model (all features)
# 2. Simple model (top 7 features)
# 3. Ultra-simple model (top 5 features)
# This helps understand if we can achieve similar performance with fewer features

# Import additional preprocessing components needed for feature subset models
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

print("\n" + "="*60)
print("MULTI-MODEL COMPARISON")
print("="*60)

# MODEL 2 (Top 7 Features)
# Based on feature importance analysis, build a simpler Random Forest model
# Using same features as decision tree for consistency
features_top7 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class',
    'Inflight entertainment', 'Customer Type', 'Leg room service'
]

print("\n[Model 2] Training Random Forest with Top 7 Features...")
# Create subset of training data with only the top 7 features
X_train_7 = X_train_full[features_top7].copy()

# Define a specialized pipeline for these 7 features
# Can't use the main preprocessor because it expects all original features
prep_7 = ColumnTransformer(transformers=[
    # Numerical features: impute missing values, then scale
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())]),
     ['Online boarding', 'Inflight wifi service', 'Inflight entertainment', 'Leg room service']),
    # Ordinal feature: encode Class as ordered categories
    ('ord', Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Eco', 'Eco Plus', 'Business']]))]),
     ['Class']),
    # Categorical features: one-hot encode with drop_first to avoid multicollinearity
    ('cat', Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))]),
     ['Type of Travel', 'Customer Type'])
])

# Create Random Forest pipeline for top 7 features
# Use same hyperparameters as full model for fair comparison
rf_7 = Pipeline(steps=[('preprocessor', prep_7),
                       ('classifier', RandomForestClassifier(
                           n_estimators=100, max_depth=15, min_samples_split=20,
                           min_samples_leaf=10, random_state=42, n_jobs=-1))])

# Evaluate using same cross-validation approach
print("Running 5-Fold CV on Top-7-Features Random Forest...")
scores_acc_7 = cross_val_score(rf_7, X_train_7, y_train_full, cv=cv, scoring='accuracy')
scores_auc_7 = cross_val_score(rf_7, X_train_7, y_train_full, cv=cv, scoring='roc_auc')

print(f"Top-7-Features RF Accuracy: {scores_acc_7.mean():.4f}")
print(f"Top-7-Features RF AUC:      {scores_auc_7.mean():.4f}")
print("-" * 30)

# MODEL 3 (Top 5 Features)
# Even simpler model to test the limits of feature reduction
features_top5 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class', 'Inflight entertainment'
]

print("\n[Model 3] Training Random Forest with Top 5 Features...")
X_train_5 = X_train_full[features_top5].copy()

# Define pipeline for top 5 features
prep_5 = ColumnTransformer(transformers=[
    # Numerical features: same approach as above
    ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())]),
     ['Online boarding', 'Inflight wifi service', 'Inflight entertainment']),
    # Ordinal feature: Class encoding
    ('ord', Pipeline(steps=[('encoder', OrdinalEncoder(categories=[['Eco', 'Eco Plus', 'Business']]))]),
     ['Class']),
    # Categorical feature: only Type of Travel in this subset
    ('cat', Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))]),
     ['Type of Travel'])
])

# Create Random Forest pipeline for top 5 features
rf_5 = Pipeline(steps=[('preprocessor', prep_5),
                       ('classifier', RandomForestClassifier(
                           n_estimators=100, max_depth=15, min_samples_split=20,
                           min_samples_leaf=10, random_state=42, n_jobs=-1))])

print("Running 5-Fold CV on Top-5-Features Random Forest...")
scores_acc_5 = cross_val_score(rf_5, X_train_5, y_train_full, cv=cv, scoring='accuracy')
scores_auc_5 = cross_val_score(rf_5, X_train_5, y_train_full, cv=cv, scoring='roc_auc')

print(f"Top-5-Features RF Accuracy: {scores_acc_5.mean():.4f}")
print(f"Top-5-Features RF AUC:      {scores_auc_5.mean():.4f}")
print("-" * 30)

print("\n" + "="*60)
print("RANDOM FOREST MODEL ANALYSIS COMPLETE!")
print("="*60)
print(f"\nModel Performance Comparison:")
print(f"  Full Model (All Features):")
print(f"    - Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"    - AUC:      {cv_results['test_roc_auc'].mean():.4f}")
print(f"\n  Top-7-Features Model:")
print(f"    - Accuracy: {scores_acc_7.mean():.4f}")
print(f"    - AUC:      {scores_auc_7.mean():.4f}")
print(f"\n  Top-5-Features Model:")
print(f"    - Accuracy: {scores_acc_5.mean():.4f}")
print(f"    - AUC:      {scores_auc_5.mean():.4f}")
print(f"\nKey Insights:")
print(f"  - Random Forest provides ensemble learning with {rf_model.n_estimators} trees")
print(f"  - Feature importance shows consistency across trees (with std deviation)")
print(f"  - Subgroup analysis reveals different priorities by customer segment")
print(f"  - Model comparison shows performance vs complexity trade-offs")
print(f"\nGenerated Visualizations:")
print(f"  ✓ rf_feature_importance.png")
print(f"  ✓ rf_feature_importance_with_std.png (Random Forest specific)")
print(f"  ✓ rf_subgroup_class_comparison.png")
print(f"  ✓ rf_subgroup_type_comparison.png")
print("\n" + "="*60)

