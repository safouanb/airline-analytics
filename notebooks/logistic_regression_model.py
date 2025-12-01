# -*- coding: utf-8 -*-
"""Logistic Regression Model for Airline Satisfaction Prediction

This script trains Logistic Regression classifiers to predict customer satisfaction.
Logistic Regression is a linear model that uses a logistic function to model
binary classification problems. It provides interpretable coefficients and
probability estimates.

Created by: Ayse Tugay
Enhanced with: Multi-model comparison and visualizations
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Import our custom preprocessing functions
# These handle data loading and feature transformation consistently with other models
from preprocessing import load_data, get_preprocessor

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
# Load the training dataset from the data/raw directory
# Following same approach as decision tree and random forest models
# Focus on cross-validation instead of test set evaluation
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
# STEP 3: MODEL 1 - FULL LOGISTIC REGRESSION (ALL FEATURES)
# ============================================================================
# Logistic Regression with all available features
# Key hyperparameters explained:
# - max_iter=1000: Maximum iterations for convergence (default 100 often insufficient)
# - class_weight="balanced": Automatically adjusts weights for imbalanced classes
# - solver="liblinear": Good for small-medium datasets, handles L1/L2 regularization
# - StandardScaler with_mean=False: Required for sparse matrices from preprocessing

print("\nConfiguring Logistic Regression (Full Model)...")
log_reg_full = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("scaler", StandardScaler(with_mean=False)),  # Essential for logistic regression
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

# ============================================================================
# STEP 4: CROSS-VALIDATION FOR FULL MODEL
# ============================================================================
# Use 5-fold cross-validation to estimate model performance
# Stratified k-fold maintains the same ratio of satisfied/dissatisfied in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold cross-validation for Full Model...")
cv_results_full = cross_validate(
    log_reg_full, X_train_full, y_train_full,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"--- FULL MODEL 5-Fold Cross-Validation Results ---")
print(f"Mean Accuracy: {cv_results_full['test_accuracy'].mean():.4f}")
print(f"Mean AUC:      {cv_results_full['test_roc_auc'].mean():.4f}")
print(f"Mean F1-Macro: {cv_results_full['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 5: MODEL 2 - TOP 7 FEATURES LOGISTIC REGRESSION
# ============================================================================
# Based on feature importance from decision tree and random forest analysis
# Using same top features as other models for consistency
features_top7 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class',
    'Inflight entertainment', 'Customer Type', 'Leg room service'
]

print("\n[Model 2] Training Logistic Regression with Top 7 Features...")
# Create subset of training data with only the top 7 features
X_train_7 = X_train_full[features_top7].copy()

# Define a specialized pipeline for these 7 features
# Can't use the main preprocessor because it expects all original features
prep_7 = ColumnTransformer(transformers=[
    # Numerical features: impute missing values, then scale (crucial for logistic regression)
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

# Create Logistic Regression pipeline for top 7 features
log_reg_7 = Pipeline(steps=[
    ("preprocessor", prep_7),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

print("Running 5-Fold CV on Top-7-Features Logistic Regression...")
cv_results_7 = cross_validate(
    log_reg_7, X_train_7, y_train_full,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"Top-7-Features LR Accuracy: {cv_results_7['test_accuracy'].mean():.4f}")
print(f"Top-7-Features LR AUC:      {cv_results_7['test_roc_auc'].mean():.4f}")
print(f"Top-7-Features LR F1-Macro: {cv_results_7['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 6: MODEL 3 - TOP 5 FEATURES LOGISTIC REGRESSION
# ============================================================================
# Even simpler model to test the limits of feature reduction
# Consistent with decision tree and random forest analysis
features_top5 = [
    'Online boarding', 'Inflight wifi service', 'Type of Travel', 'Class', 'Inflight entertainment'
]

print("\n[Model 3] Training Logistic Regression with Top 5 Features...")
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

# Create Logistic Regression pipeline for top 5 features
log_reg_5 = Pipeline(steps=[
    ("preprocessor", prep_5),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

print("Running 5-Fold CV on Top-5-Features Logistic Regression...")
cv_results_5 = cross_validate(
    log_reg_5, X_train_5, y_train_full,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"Top-5-Features LR Accuracy: {cv_results_5['test_accuracy'].mean():.4f}")
print(f"Top-5-Features LR AUC:      {cv_results_5['test_roc_auc'].mean():.4f}")
print(f"Top-5-Features LR F1-Macro: {cv_results_5['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 7: FEATURE IMPORTANCE ANALYSIS (COEFFICIENTS)
# ============================================================================
# Train the full model to analyze feature importance through coefficients
# Logistic regression coefficients show the log-odds impact of each feature
print("\nTraining full model for coefficient analysis...")
log_reg_full.fit(X_train_full, y_train_full)

# Reconstruct feature names after preprocessing
numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()
ordinal_features = ['Class']
nominal_features = ['Gender', 'Customer Type', 'Type of Travel']
cat_names = log_reg_full.named_steps['preprocess'].named_transformers_['cat']['encoder'].get_feature_names_out(nominal_features).tolist()

# Combine all feature names in the same order as the model sees them
feature_names = numerical_features + ordinal_features + cat_names

# Extract coefficients from the trained Logistic Regression
# Positive coefficients increase probability of satisfaction
# Negative coefficients decrease probability of satisfaction
coefficients = log_reg_full.named_steps['model'].coef_[0]
coeff_importance = pd.Series(np.abs(coefficients), index=feature_names).sort_values(ascending=False)

# Plot the top 10 most important features (by absolute coefficient value)
plt.figure(figsize=(10, 6))
coeff_importance.head(10).plot(kind='barh', color='#2E8B57')
plt.title('Top 10 Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Absolute Coefficient Value (Impact on Log-Odds)')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.savefig('lr_feature_importance.png')
print("Saved lr_feature_importance.png")

# ============================================================================
# STEP 8: MODEL PERFORMANCE COMPARISON VISUALIZATION
# ============================================================================
# Create a comparison chart showing all three models' performance
models = ['Full Model\n(All Features)', 'Top 7 Features', 'Top 5 Features']
accuracies = [
    cv_results_full['test_accuracy'].mean(),
    cv_results_7['test_accuracy'].mean(),
    cv_results_5['test_accuracy'].mean()
]
aucs = [
    cv_results_full['test_roc_auc'].mean(),
    cv_results_7['test_roc_auc'].mean(),
    cv_results_5['test_roc_auc'].mean()
]

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax1.bar(models, accuracies, color=['#2E8B57', '#4682B4', '#CD853F'])
ax1.set_title('Accuracy Comparison (5-Fold CV)')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.8, 0.9)
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

# AUC comparison
ax2.bar(models, aucs, color=['#2E8B57', '#4682B4', '#CD853F'])
ax2.set_title('AUC Comparison (5-Fold CV)')
ax2.set_ylabel('AUC')
ax2.set_ylim(0.9, 0.95)
for i, v in enumerate(aucs):
    ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('lr_model_comparison.png')
print("Saved lr_model_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL ANALYSIS COMPLETE!")
print("="*60)
print(f"\nModel Performance Comparison:")
print(f"  Full Model (All Features):")
print(f"    - Accuracy: {cv_results_full['test_accuracy'].mean():.4f}")
print(f"    - AUC:      {cv_results_full['test_roc_auc'].mean():.4f}")
print(f"    - F1-Macro: {cv_results_full['test_f1_macro'].mean():.4f}")
print(f"\n  Top-7-Features Model:")
print(f"    - Accuracy: {cv_results_7['test_accuracy'].mean():.4f}")
print(f"    - AUC:      {cv_results_7['test_roc_auc'].mean():.4f}")
print(f"    - F1-Macro: {cv_results_7['test_f1_macro'].mean():.4f}")
print(f"\n  Top-5-Features Model:")
print(f"    - Accuracy: {cv_results_5['test_accuracy'].mean():.4f}")
print(f"    - AUC:      {cv_results_5['test_roc_auc'].mean():.4f}")
print(f"    - F1-Macro: {cv_results_5['test_f1_macro'].mean():.4f}")
print(f"\nKey Insights:")
print(f"  - Logistic Regression provides interpretable linear decision boundary")
print(f"  - Feature coefficients show log-odds impact on satisfaction probability")
print(f"  - Model performs well even with reduced feature sets")
print(f"  - class_weight='balanced' handles class imbalance automatically")
print(f"\nGenerated Visualizations:")
print(f"  ✓ lr_feature_importance.png")
print(f"  ✓ lr_model_comparison.png")
print("\n" + "="*60)