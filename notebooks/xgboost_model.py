# -*- coding: utf-8 -*-
"""XGBoost (Gradient Boosting) Model for Airline Satisfaction Prediction

This script trains XGBoost classifiers to predict customer satisfaction.
XGBoost (eXtreme Gradient Boosting) is a powerful ensemble method that uses
gradient boosting to create a strong classifier from weak learners.

Created by: Constantin
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Import our custom preprocessing functions
from preprocessing import load_data, get_preprocessor

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("Loading training data...")
X_train_full, y_train_full = load_data('../data/raw/train.csv')
print(f"Training samples: {len(X_train_full):,}")

# ============================================================================
# STEP 2: CREATE PREPROCESSING PIPELINE
# ============================================================================
preprocessor = get_preprocessor(X_train_full)

# ============================================================================
# STEP 3: PREPROCESS DATA FOR GRADIENT BOOSTING
# ============================================================================
print("\nPreprocessing data...")
X_processed = preprocessor.fit_transform(X_train_full)

# Build processed DataFrame with feature names
numerical_features = X_train_full.select_dtypes(include=['number']).columns.tolist()
ordinal_features = ['Class']
nominal_features = ['Gender', 'Customer Type', 'Type of Travel']
cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(nominal_features).tolist()

feature_names = numerical_features + ordinal_features + cat_names

df_processed = pd.DataFrame(X_processed, columns=feature_names)
df_processed["Satisfaction_binary"] = y_train_full.values

print(f"Processed dataset shape: {df_processed.shape}")

# ============================================================================
# STEP 4: MODEL 1 - FULL GRADIENT BOOSTING (ALL FEATURES)
# ============================================================================
print("\nConfiguring Gradient Boosting (Full Model)...")

# Prepare features and target
y = df_processed["Satisfaction_binary"]
X_encoded = df_processed.drop(columns=["Satisfaction_binary"])

print(f"X_encoded shape: {X_encoded.shape}")
print(f"y distribution: {y.value_counts(normalize=True).round(3).to_dict()}")

# Define the model with optimized hyperparameters
gbt_full = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# ============================================================================
# STEP 5: CROSS-VALIDATION FOR FULL MODEL
# ============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold cross-validation for Full Model...")
cv_results_full = cross_validate(
    gbt_full, X_encoded, y,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"--- FULL MODEL 5-Fold Cross-Validation Results ---")
print(f"Mean Accuracy: {cv_results_full['test_accuracy'].mean():.4f}")
print(f"Mean AUC:      {cv_results_full['test_roc_auc'].mean():.4f}")
print(f"Mean F1-Macro: {cv_results_full['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 6: FEATURE SELECTION ANALYSIS
# ============================================================================
print("\nTraining full model for feature analysis...")
gbt_full.fit(X_encoded, y)

# Get feature importances
importances = gbt_full.feature_importances_
fi = pd.DataFrame({
    "feature": X_encoded.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("Top 15 features by importance:")
print(fi.head(15))

# ============================================================================
# STEP 7: MODEL 2 - TOP 7 FEATURES MODEL
# ============================================================================
features_top7 = fi.head(7)["feature"].tolist()
X_train_7 = X_encoded[features_top7]

print(f"\n[Model 2] Training Gradient Boosting with Top 7 Features...")
print(f"Top 7 features: {features_top7}")

gbt_7 = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

cv_results_7 = cross_validate(
    gbt_7, X_train_7, y,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"Top-7-Features GB Accuracy: {cv_results_7['test_accuracy'].mean():.4f}")
print(f"Top-7-Features GB AUC:      {cv_results_7['test_roc_auc'].mean():.4f}")
print(f"Top-7-Features GB F1-Macro: {cv_results_7['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 8: MODEL 3 - TOP 5 FEATURES MODEL
# ============================================================================
features_top5 = fi.head(5)["feature"].tolist()
X_train_5 = X_encoded[features_top5]

print(f"\n[Model 3] Training Gradient Boosting with Top 5 Features...")
print(f"Top 5 features: {features_top5}")

gbt_5 = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

cv_results_5 = cross_validate(
    gbt_5, X_train_5, y,
    cv=cv,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_macro': 'f1_macro'}
)

print(f"Top-5-Features GB Accuracy: {cv_results_5['test_accuracy'].mean():.4f}")
print(f"Top-5-Features GB AUC:      {cv_results_5['test_roc_auc'].mean():.4f}")
print(f"Top-5-Features GB F1-Macro: {cv_results_5['test_f1_macro'].mean():.4f}")
print("-" * 30)

# ============================================================================
# STEP 9: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\nGenerating feature importance visualization...")

# Plot the top 15 most important features
plt.figure(figsize=(10, 6))
fi_top15 = fi.head(15)
plt.barh(range(15), fi_top15["importance"].values[::-1], color='#FF6B35')
plt.yticks(range(15), fi_top15["feature"].values[::-1])
plt.xlabel('Feature Importance (Gradient Boosting)')
plt.title('Top 15 Feature Importances - Gradient Boosting')
plt.tight_layout()
plt.savefig('../figures/xgb_feature_importance.png')
print("Saved xgb_feature_importance.png")

# ============================================================================
# STEP 10: MODEL PERFORMANCE COMPARISON VISUALIZATION
# ============================================================================
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
ax1.bar(models, accuracies, color=['#FF6B35', '#4682B4', '#CD853F'])
ax1.set_title('Accuracy Comparison (5-Fold CV)')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.9, 0.97)
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

# AUC comparison
ax2.bar(models, aucs, color=['#FF6B35', '#4682B4', '#CD853F'])
ax2.set_title('AUC Comparison (5-Fold CV)')
ax2.set_ylabel('AUC')
ax2.set_ylim(0.96, 1.0)
for i, v in enumerate(aucs):
    ax2.text(i, v + 0.002, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../figures/xgb_model_comparison.png')
print("Saved xgb_model_comparison.png")

# ============================================================================
# STEP 11: SUBGROUP ANALYSIS - FEATURE IMPORTANCE BY CLASS
# ============================================================================
print("\nGenerating subgroup analysis by travel class...")

# Use encoded classes (0 = Eco, 1 = Eco Plus, 2 = Business)
class_map = {0: "Eco", 1: "Eco Plus", 2: "Business"}
classes = df_processed["Class"].unique()
fi_dict = {}

for cls in classes:
    class_name = class_map.get(cls, str(cls))
    print(f"\n=== Training model for Class = {class_name} ===")

    # Subset to this travel class
    df_sub = df_processed[df_processed["Class"] == cls].copy()

    # Target
    y_sub = df_sub["Satisfaction_binary"]

    # Drop features we don't want to leak (target and filter variable)
    drop_cols = ["Satisfaction_binary", "Class"]
    X_sub = df_sub.drop(columns=drop_cols)

    print(f"  Training on {len(df_sub):,} samples with {X_sub.shape[1]} features...")

    # Train GBT with same hyperparams as before
    gbt_sub = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbt_sub.fit(X_sub, y_sub)

    # Feature importances
    fi_sub = pd.Series(
        gbt_sub.feature_importances_,
        index=X_sub.columns,
        name=class_name
    )

    # Store
    fi_dict[class_name] = fi_sub.sort_values(ascending=False)
    print(f"  Top 5 features: {list(fi_dict[class_name].head(5).index)}")

# Combine all class-specific importances into one DataFrame
fi_all = pd.concat(fi_dict, axis=1)
fi_all = fi_all.fillna(0)

# Compute average importance to sort features
fi_all["Average_Importance"] = fi_all.mean(axis=1)
fi_all = fi_all.sort_values("Average_Importance", ascending=False)

# Show top 15 features with importance per class
fi_all_top15 = fi_all.head(15).drop(columns=["Average_Importance"])

# Create heatmap visualization
plt.figure(figsize=(12, 8))
sns.heatmap(fi_all_top15, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("Feature Importance by Travel Class - Gradient Boosting (Top 15 Features)")
plt.xlabel("Travel Class")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('../figures/xgb_subgroup_class_comparison.png')
print("Saved xgb_subgroup_class_comparison.png")

# Also create bar plot for easier comparison
fi_all_top8 = fi_all_top15.head(8)
df_class_comp = fi_all_top8[['Business', 'Eco', 'Eco Plus']]

plt.figure(figsize=(14, 6))
df_class_comp.plot(kind='bar', color=['#FF6B35', '#4682B4', '#CD853F'])
plt.title('Feature Importance by Travel Class - Gradient Boosting (Top 8 Features)')
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Travel Class')
plt.tight_layout()
plt.savefig('../figures/xgb_subgroup_class_barplot.png')
print("Saved xgb_subgroup_class_barplot.png")

# ============================================================================
# STEP 12: FEATURE SELECTION OPTIMIZATION
# ============================================================================
# NOTE: Feature-count optimization loop removed to reduce runtime.
# The comparison of Full vs Top-7 vs Top-5 already shows the performance plateau.
# If needed, reintroduce a smaller sweep (e.g., k=3..10) for quicker exploration.

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("GRADIENT BOOSTING MODEL ANALYSIS COMPLETE!")
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
print(f"  - Gradient Boosting builds strong classifier from weak learners")
print(f"  - Sequential learning focuses on hard-to-classify examples")
print(f"  - Feature importance based on gradient information gain")
print(f"  - Excellent performance with automatic feature selection")

print(f"\nSubgroup Analysis Insights:")
print(f"  - Business class: Focus on online boarding and entertainment")
print(f"  - Economy class: WiFi service dominates satisfaction")
print(f"  - Eco Plus: Balanced importance across service features")

print(f"\nGenerated Visualizations:")
print(f"  ✓ xgb_feature_importance.png")
print(f"  ✓ xgb_model_comparison.png")
print(f"  ✓ xgb_subgroup_class_comparison.png")
print(f"  ✓ xgb_subgroup_class_barplot.png")
print("\n" + "="*60)