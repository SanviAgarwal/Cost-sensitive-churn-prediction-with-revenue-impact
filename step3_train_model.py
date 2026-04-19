# =============================================================
# STEP 3 — TRAIN THE MACHINE LEARNING MODELS
# =============================================================
# What this file does:
#   - Trains a simple baseline model (Logistic Regression)
#   - Trains a powerful model (XGBoost)
#   - Compares them with proper metrics (not just accuracy)
#   - Shows a confusion matrix for each
#   - Saves the best model for later steps
# =============================================================
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. LOAD THE PREPARED DATA FROM STEP 2
# --------------------------------------------------
with open("outputs/prepared_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

print("=" * 55)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 55)
print()

# ==========================================================
# MODEL 1: LOGISTIC REGRESSION (THE BASELINE)
# ==========================================================
# Think of this as: drawing a straight line to separate
# "will churn" from "will stay"
# Simple, fast, interpretable
# class_weight='balanced' → tells the model to care more about
# catching churners, because they're the minority

print("Training Model 1: Logistic Regression (baseline)...")
lr_model = LogisticRegression(
    class_weight="balanced",  # Handle the 26% imbalance
    max_iter=1000,            # Give it enough tries to converge
    random_state=42
)
lr_model.fit(X_train, y_train)

# Get predictions on the TEST SET (data model never saw)
lr_preds = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]  # Probability score 0-1

lr_auc = roc_auc_score(y_test, lr_probs)
print(f"  Done. AUC Score: {lr_auc:.4f}")
print()

# ==========================================================
# MODEL 2: XGBOOST (THE CHAMPION)
# ==========================================================
# Think of this as: many decision trees working together
# Like asking 300 different experts and taking the majority vote
# Much more powerful than a single model

print("Training Model 2: XGBoost (champion model)...")

# Count how many more "stayed" customers there are vs "churned"
# We tell XGBoost to weight churners more heavily
scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=300,         # 300 trees
    learning_rate=0.05,       # How much each tree "corrects" the previous
    max_depth=5,              # How deep each tree can go
    scale_pos_weight=scale,   # Handle imbalance
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_probs)
print(f"  Done. AUC Score: {xgb_auc:.4f}")
print()

# --------------------------------------------------
# 2. COMPARE RESULTS
# --------------------------------------------------
print("=" * 55)
print("RESULTS COMPARISON")
print("=" * 55)
print()

for name, preds, probs, auc in [
    ("Logistic Regression", lr_preds, lr_probs, lr_auc),
    ("XGBoost",             xgb_preds, xgb_probs, xgb_auc),
]:
    print(f"--- {name} ---")
    print(f"AUC Score: {auc:.4f}")
    print()
    # classification_report gives precision, recall, f1 for each class
    print(classification_report(y_test, preds,
                                target_names=["Stayed (0)", "Churned (1)"]))

# --------------------------------------------------
# 3. UNDERSTANDING THE METRICS
# --------------------------------------------------
print("=" * 55)
print("WHAT DO THESE NUMBERS MEAN?")
print("=" * 55)
print()
print("AUC (Area Under Curve):")
print("  0.5 = random guessing | 1.0 = perfect model")
print(f"  XGBoost got {xgb_auc:.3f} — that's quite good")
print()
print("Precision (for Churned class):")
print("  Of everyone the model flagged as 'will churn',")
print("  what % actually did churn?")
print()
print("Recall (for Churned class):")
print("  Of everyone who actually churned,")
print("  what % did the model catch?")
print()
print("For this business problem, RECALL is more important.")
print("Missing a churner (low recall) costs ~₹500 in lost CLV.")
print("A false alarm (low precision) costs only ~₹20 in wasted offer.")
print()

# --------------------------------------------------
# 4. PLOT CONFUSION MATRICES SIDE BY SIDE
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrices — What the model got right and wrong",
             fontsize=14, fontweight="bold")

for ax, name, preds in [
    (axes[0], "Logistic Regression", lr_preds),
    (axes[1], "XGBoost",             xgb_preds),
]:
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap="Blues", cbar=False,
        xticklabels=["Predicted: Stayed", "Predicted: Churned"],
        yticklabels=["Actual: Stayed", "Actual: Churned"]
    )
    ax.set_title(name)

    # Print what each cell means below the chart
    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(
        f"Correct stays (TN): {tn}   |   "
        f"Wrong flags (FP): {fp}\n"
        f"Missed churners (FN): {fn}   |   "
        f"Caught churners (TP): {tp}"
    )

plt.tight_layout()
plt.savefig("outputs/step3_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 5. PLOT ROC AND PR CURVES
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Quality Curves", fontsize=14, fontweight="bold")

# ROC Curve — how well does the model separate churners from stayers?
RocCurveDisplay.from_predictions(
    y_test, lr_probs, ax=axes[0], name="Logistic Regression"
)
RocCurveDisplay.from_predictions(
    y_test, xgb_probs, ax=axes[0], name="XGBoost"
)
axes[0].set_title("ROC Curve (higher = better)")
axes[0].plot([0, 1], [0, 1], "k--", label="Random guess")

# PR Curve — better for imbalanced data
PrecisionRecallDisplay.from_predictions(
    y_test, lr_probs, ax=axes[1], name="Logistic Regression"
)
PrecisionRecallDisplay.from_predictions(
    y_test, xgb_probs, ax=axes[1], name="XGBoost"
)
axes[1].set_title("Precision-Recall Curve (use this for imbalanced data)")

plt.tight_layout()
plt.savefig("outputs/step3_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 6. SAVE THE BEST MODEL
# --------------------------------------------------
with open("outputs/best_model.pkl", "wb") as f:
    pickle.dump({
        "model": xgb_model,
        "model_name": "XGBoost",
        "X_test": X_test,
        "y_test": y_test,
        "xgb_probs": xgb_probs,
        "feature_names": data["feature_names"]
    }, f)

print("Best model (XGBoost) saved to: outputs/best_model.pkl")
print()
# print("Step 3 complete! Run step4_cost_matrix.py next.")