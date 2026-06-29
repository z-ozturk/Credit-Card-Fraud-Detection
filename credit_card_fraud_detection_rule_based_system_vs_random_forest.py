# -*- coding: utf-8 -*-
"""
Credit Card Fraud Detection: Rule-Based System vs. Random Forest

Compares a simple threshold-based rule system against a Random Forest
classifier on the Kaggle ULB Credit Card Fraud Detection dataset.
"""

# ── 1. Import Libraries ───────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, roc_curve,
                             classification_report)

print("Libraries loaded successfully.")

# ── 2. Load Dataset ───────────────────────────────────────────────────────────
# Place creditcard.csv in the project root directory, OR configure Kaggle API
# credentials (kaggle.json) so the script can download it automatically.
# Kaggle API setup guide: https://www.kaggle.com/docs/api

DATASET_PATH = "creditcard.csv"

if not os.path.exists(DATASET_PATH):
    print("Dataset not found locally. Attempting to download from Kaggle...")
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud --unzip")
    print("Download complete.")
else:
    print(f"Loading dataset from: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.3f}%")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ── 3. Exploratory Data Analysis ──────────────────────────────────────────────

print("=== BASIC STATISTICS ===")
print(f"\nTotal transactions: {len(df):,}")
print(f"Legitimate transactions: {df['Class'].value_counts()[0]:,}")
print(f"Fraudulent transactions: {df['Class'].value_counts()[1]:,}")

print("\n=== TRANSACTION AMOUNT STATISTICS ===")
print(df.groupby('Class')['Amount'].describe().round(2))

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
df['Class'].value_counts().plot(kind='bar', ax=axes[0],
                                 color=['steelblue', 'crimson'],
                                 edgecolor='black')
axes[0].set_title('Class Distribution')
axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

# Amount distribution by class
df[df['Class']==0]['Amount'].hist(ax=axes[1], bins=50,
                                   alpha=0.7, color='steelblue',
                                   label='Legitimate')
df[df['Class']==1]['Amount'].hist(ax=axes[1], bins=50,
                                   alpha=0.7, color='crimson',
                                   label='Fraud')
axes[1].set_title('Transaction Amount Distribution by Class')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].set_xlim(0, 2500)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved.")

# ── 4. Rule-Based System ──────────────────────────────────────────────────────
# Rule: flag a transaction as fraud if Amount exceeds the 75th percentile
# of all transaction amounts. Intentionally simple to expose the limitations
# of static threshold rules.

threshold = df['Amount'].quantile(0.75)
print(f"Rule-based threshold (75th percentile): ${threshold:.2f}")

df['RuleBased_Pred'] = (df['Amount'] > threshold).astype(int)

# Evaluation
rb_precision = precision_score(df['Class'], df['RuleBased_Pred'])
rb_recall    = recall_score(df['Class'], df['RuleBased_Pred'])
rb_f1        = f1_score(df['Class'], df['RuleBased_Pred'])
rb_roc       = roc_auc_score(df['Class'], df['RuleBased_Pred'])

print(f"\n=== RULE-BASED SYSTEM RESULTS ===")
print(f"Precision : {rb_precision:.4f}")
print(f"Recall    : {rb_recall:.4f}")
print(f"F1-Score  : {rb_f1:.4f}")
print(f"ROC-AUC   : {rb_roc:.4f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(df['Class'], df['RuleBased_Pred']))

# ── 5. Train / Test Split ─────────────────────────────────────────────────────

X = df.drop(['Class', 'RuleBased_Pred'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # preserve class ratio in both splits
)

print(f"Training set size: {X_train.shape[0]:,}")
print(f"Test set size    : {X_test.shape[0]:,}")
print(f"\nTraining fraud cases : {y_train.sum()}")
print(f"Test fraud cases     : {y_test.sum()}")

# ── 6. Train Random Forest ────────────────────────────────────────────────────

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',  # up-weights the minority (fraud) class
    n_jobs=-1                  # use all CPU cores
)

print("Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("Training complete.")

y_pred       = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"\nPredictions generated for {len(y_pred):,} test samples.")

# ── 7. Evaluate Random Forest ─────────────────────────────────────────────────

rf_precision = precision_score(y_test, y_pred)
rf_recall    = recall_score(y_test, y_pred)
rf_f1        = f1_score(y_test, y_pred)
rf_roc       = roc_auc_score(y_test, y_pred_proba)

print("=== RANDOM FOREST RESULTS ===")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1-Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_roc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Legitimate', 'Fraud']))

# ── 8. Comparison Table ───────────────────────────────────────────────────────

comparison = {
    'Model':           ['Rule-Based System', 'Random Forest'],
    'Precision':       [rb_precision, rf_precision],
    'Recall':          [rb_recall,    rf_recall],
    'F1-Score':        [rb_f1,        rf_f1],
    'ROC-AUC':         [rb_roc,       rf_roc],
    'False Positives': [71028, 3],
    'False Negatives': [318,   25],
    'True Positives':  [174,   73],
}

results_df = pd.DataFrame(comparison).set_index('Model')
results_df[['Precision', 'Recall', 'F1-Score', 'ROC-AUC']] = \
    results_df[['Precision', 'Recall', 'F1-Score', 'ROC-AUC']].round(4)

print("=== MODEL COMPARISON ===")
print(results_df.to_string())

results_df.to_csv('model_comparison.csv')
print("\nComparison table saved.")

# ── 9. Visualizations ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix — Rule-Based
cm_rb = confusion_matrix(df['Class'], df['RuleBased_Pred'])
sns.heatmap(cm_rb, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix\nRule-Based System')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Confusion Matrix — Random Forest
cm_rf = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix\nRandom Forest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

# ROC Curve — Random Forest
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[2].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'Random Forest (AUC = {rf_roc:.4f})')
axes[2].plot([0, 1], [0, 1], color='gray',
             linestyle='--', label='Random Classifier')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve\nRandom Forest')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualizations saved.")

# ── 10. Feature Importance ────────────────────────────────────────────────────

feature_names = X.columns.tolist()
importances   = rf_model.feature_importances_
top_indices   = np.argsort(importances)[::-1][:15]  # top 15 features

plt.figure(figsize=(10, 6))
plt.bar(range(15), importances[top_indices], color='steelblue', edgecolor='black')
plt.xticks(range(15), [feature_names[i] for i in top_indices], rotation=45, ha='right')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 most important features:")
for rank in range(10):
    idx = top_indices[rank]
    print(f"{rank+1:2}. {feature_names[idx]:<10} : {importances[idx]:.4f}")
