# =============================================================
# STEP 5 — SHAP: WHY DID THE MODEL FLAG THIS CUSTOMER?
# =============================================================
# What this file does:
#   - Uses SHAP to explain WHY the model makes each prediction
#   - Shows which features matter most overall
#   - Shows exactly why one specific high-risk customer was flagged
#   - This is what you show to business teams
# =============================================================
# SHAP = SHapley Additive exPlanations
# Plain English: "For this customer, feature X pushed their risk
# score UP by 0.15, and feature Y pushed it DOWN by 0.08..."
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. LOAD MODEL AND DATA
# --------------------------------------------------
with open("outputs/best_model.pkl", "rb") as f:
    saved = pickle.load(f)
with open("outputs/cost_analysis.pkl", "rb") as f:
    cost = pickle.load(f)

model         = saved["model"]
X_test        = saved["X_test"]
y_test        = saved["y_test"]
xgb_probs     = saved["xgb_probs"]
feature_names = saved["feature_names"]
best_threshold = cost["best_threshold"]

print("=" * 55)
print("STEP 5: SHAP EXPLAINABILITY")
print("=" * 55)
print()
print("Calculating SHAP values... (this takes ~10-20 seconds)")

# --------------------------------------------------
# 2. CALCULATE SHAP VALUES
# --------------------------------------------------
# TreeExplainer is the fast version for tree-based models like XGBoost
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("Done.")
print()

# --------------------------------------------------
# 3. GLOBAL FEATURE IMPORTANCE CHART
# --------------------------------------------------
# This shows: across ALL customers in the test set,
# which features pushed predictions up or down the most?

print("Generating global feature importance chart...")

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("Top 15 Features That Drive Churn Predictions\n(longer bar = more important)", 
          fontsize=13)
plt.tight_layout()
plt.savefig("outputs/step5_global_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 4. SHAP SUMMARY PLOT (BEESWARM)
# --------------------------------------------------
# This is richer — shows both direction and magnitude
# Red dots = high value of that feature → pushed prediction toward churn
# Blue dots = low value → pushed prediction away from churn

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values, X_test,
    max_display=15,
    show=False
)
plt.title("How Each Feature Affects Churn Risk\n(red = increases risk, blue = decreases risk)",
          fontsize=13)
plt.tight_layout()
plt.savefig("outputs/step5_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 5. EXPLAIN ONE SPECIFIC HIGH-RISK CUSTOMER
# --------------------------------------------------
# Find the customer with the highest predicted churn probability
high_risk_idx = np.argmax(xgb_probs)
high_risk_prob = xgb_probs[high_risk_idx]
high_risk_features = X_test.iloc[high_risk_idx]
actual_label = y_test.iloc[high_risk_idx]

print("=" * 55)
print("DEEP DIVE: HIGHEST RISK CUSTOMER")
print("=" * 55)
print(f"Predicted churn probability: {high_risk_prob:.1%}")
print(f"Actually churned:            {'YES' if actual_label == 1 else 'NO'}")
print()
print("Their profile:")
# Show the most informative columns
display_cols = {
    "tenure": "Months with company",
    "MonthlyCharges": "Monthly bill (₹)",
    "Contract": "Contract (0=monthly, 1=annual, 2=biannual)",
    "TotalCharges": "Total paid (₹)",
    "charge_per_month_tenure": "Charge/tenure ratio"
}
for col, label in display_cols.items():
    if col in high_risk_features.index:
        print(f"  {label}: {high_risk_features[col]:.2f}")
print()

# Waterfall plot — explains this single customer's prediction
# Each bar shows how much one feature pushed the risk score up or down
plt.figure(figsize=(12, 7))
shap_explanation = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[high_risk_idx].values,
    feature_names=feature_names
)
shap.waterfall_plot(shap_explanation, max_display=12, show=False)
plt.title(
    f"Why is this customer {high_risk_prob:.1%} likely to churn?\n"
    f"Each bar shows one feature pushing the risk score up (red) or down (blue)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("outputs/step5_waterfall_single_customer.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 6. PRINT A PLAIN ENGLISH EXPLANATION
# --------------------------------------------------
# Find the top 3 features pushing this customer toward churn
shap_series = pd.Series(shap_values[high_risk_idx], index=feature_names)
top_risk_drivers = shap_series.nlargest(3)
top_protective   = shap_series.nsmallest(3)

print("WHY IS THIS CUSTOMER HIGH RISK? (top reasons):")
for feat, val in top_risk_drivers.items():
    feature_value = high_risk_features.get(feat, "N/A")
    print(f"  + {feat} = {feature_value:.2f}  →  pushed risk UP by {val:.3f}")

print()
print("FACTORS REDUCING THEIR RISK:")
for feat, val in top_protective.items():
    feature_value = high_risk_features.get(feat, "N/A")
    print(f"  - {feat} = {feature_value:.2f}  →  pushed risk DOWN by {abs(val):.3f}")

# --------------------------------------------------
# 7. SAVE SHAP VALUES FOR THE STREAMLIT APP
# --------------------------------------------------
with open("outputs/shap_values.pkl", "wb") as f:
    pickle.dump({
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "X_test": X_test,
        "feature_names": feature_names
    }, f)

# print("SHAP values saved to: outputs/shap_values.pkl")
# print("Step 5 complete! Run step6_app.py with: streamlit run step6_app.py")