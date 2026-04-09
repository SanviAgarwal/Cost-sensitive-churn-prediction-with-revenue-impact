# =============================================================
# STEP 4 — BUSINESS FRAMING: DOLLAR-VALUE CONFUSION MATRIX
# =============================================================
# What this file does:
#   - Attaches real dollar costs to each type of prediction
#   - Finds the optimal threshold (not just 0.5)
#   - Shows how much revenue your model saves vs doing nothing
#   - This is the part that impresses interviewers and managers
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle

# --------------------------------------------------
# 1. LOAD THE MODEL AND DATA
# --------------------------------------------------
with open("outputs/best_model.pkl", "rb") as f:
    saved = pickle.load(f)

model       = saved["model"]
X_test      = saved["X_test"]
y_test      = saved["y_test"]
xgb_probs   = saved["xgb_probs"]

print("=" * 55)
print("STEP 4: COST-SENSITIVE BUSINESS ANALYSIS")
print("=" * 55)
print()

# --------------------------------------------------
# 2. DEFINE THE COST MATRIX
# --------------------------------------------------
# These are the dollar values attached to each outcome.
# Adjust these to match your actual business numbers.

# How much revenue a loyal customer brings over their lifetime
CUSTOMER_LIFETIME_VALUE = 500   # USD

# Cost of a retention offer (discount, free upgrade, etc.)
RETENTION_OFFER_COST = 20       # USD

# If we correctly flag a churner and save them, we keep their CLV
# but we spend the offer cost
COST_TRUE_POSITIVE  = CUSTOMER_LIFETIME_VALUE - RETENTION_OFFER_COST  # +₹480 net gain

# If we flag someone as "will churn" but they were loyal anyway,
# we wasted the offer on someone who wasn't leaving
COST_FALSE_POSITIVE = -RETENTION_OFFER_COST     # -₹20 wasted

# If we miss a churner (say they're fine, they leave), we lose their CLV
COST_FALSE_NEGATIVE = -CUSTOMER_LIFETIME_VALUE  # -₹500 lost

# If we correctly predict someone will stay, we do nothing — no cost
COST_TRUE_NEGATIVE  = 0                         # ₹0

print("COST MATRIX (what each outcome is worth):")
print(f"  True Positive  (caught a churner):        +₹{COST_TRUE_POSITIVE}")
print(f"  True Negative  (correctly left alone):     ₹{COST_TRUE_NEGATIVE}")
print(f"  False Positive (wasted offer on loyal):    ₹{COST_FALSE_POSITIVE}")
print(f"  False Negative (missed a churner):         ₹{COST_FALSE_NEGATIVE}")
print()

# --------------------------------------------------
# 3. WHAT WOULD HAPPEN IF WE DID NOTHING?
# --------------------------------------------------
# Baseline: never intervene. Everyone who churns is just lost.
actual_churners = y_test.sum()
baseline_loss = actual_churners * CUSTOMER_LIFETIME_VALUE

print(f"Baseline (do nothing):")
print(f"  Customers who churn:  {actual_churners}")
print(f"  Revenue lost:         -₹{baseline_loss:,}")
print()

# --------------------------------------------------
# 4. FUNCTION: CALCULATE REVENUE IMPACT AT ANY THRESHOLD
# --------------------------------------------------
# "Threshold" = the probability cutoff for flagging someone as "will churn"
# Default is 0.5. But maybe 0.3 is better for business?

def calculate_revenue_impact(y_true, y_probs, threshold):
    """
    Given a threshold, predict who churns and calculate the net revenue impact.
    Returns: revenue impact in dollars, and the confusion matrix values.
    """
    # Apply threshold: if score >= threshold, predict churn (1), else 0
    y_pred = (y_probs >= threshold).astype(int)

    # Count each outcome
    tp = ((y_pred == 1) & (y_true == 1)).sum()  # Caught churners
    tn = ((y_pred == 0) & (y_true == 0)).sum()  # Correctly left alone
    fp = ((y_pred == 1) & (y_true == 0)).sum()  # Wasted offers
    fn = ((y_pred == 0) & (y_true == 1)).sum()  # Missed churners

    # Revenue impact
    revenue = (
        tp * COST_TRUE_POSITIVE +
        fp * COST_FALSE_POSITIVE +
        fn * COST_FALSE_NEGATIVE +
        tn * COST_TRUE_NEGATIVE
    )
    return revenue, tp, tn, fp, fn


# --------------------------------------------------
# 5. SWEEP THRESHOLDS FROM 0.1 TO 0.9
# --------------------------------------------------
thresholds = np.arange(0.10, 0.91, 0.01)
results = []

for t in thresholds:
    revenue, tp, tn, fp, fn = calculate_revenue_impact(y_test, xgb_probs, t)
    n_flagged = tp + fp  # How many customers we flag for intervention
    results.append({
        "threshold": round(t, 2),
        "revenue_impact": revenue,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_flagged": n_flagged,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0
    })

results_df = pd.DataFrame(results)

# Find the threshold that maximises revenue
best_idx = results_df["revenue_impact"].idxmax()
best_row = results_df.loc[best_idx]

print(f"OPTIMAL THRESHOLD (maximises revenue): {best_row['threshold']}")
print(f"  Revenue impact at this threshold:  ₹{best_row['revenue_impact']:,.0f}")
print(f"  True Positives  (caught churners): {int(best_row['tp'])}")
print(f"  False Positives (wasted offers):   {int(best_row['fp'])}")
print(f"  False Negatives (missed churners): {int(best_row['fn'])}")
print(f"  True Negatives  (correct stays):   {int(best_row['tn'])}")
print(f"  Customers flagged for action:       {int(best_row['n_flagged'])}")
print()

# Compare to default 0.5
default_revenue, *_ = calculate_revenue_impact(y_test, xgb_probs, 0.5)
improvement = best_row["revenue_impact"] - default_revenue
print(f"At default threshold (0.5):          ₹{default_revenue:,.0f}")
print(f"At optimal threshold ({best_row['threshold']}):       ₹{best_row['revenue_impact']:,.0f}")
print(f"Extra revenue from tuning threshold: +₹{improvement:,.0f}")
print()

# --------------------------------------------------
# 6. VISUALISE: REVENUE VS THRESHOLD
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Cost-Sensitive Business Analysis", fontsize=15, fontweight="bold")

# Chart 1: Revenue impact vs threshold
axes[0, 0].plot(
    results_df["threshold"], results_df["revenue_impact"],
    color="#2196F3", linewidth=2
)
axes[0, 0].axvline(
    best_row["threshold"], color="#F44336", linestyle="--",
    label=f"Optimal: {best_row['threshold']}"
)
axes[0, 0].axvline(
    0.5, color="#FF9800", linestyle="--", alpha=0.7,
    label="Default: 0.50"
)
axes[0, 0].fill_between(
    results_df["threshold"], results_df["revenue_impact"],
    alpha=0.1, color="#2196F3"
)
axes[0, 0].set_title("Revenue Impact vs Threshold")
axes[0, 0].set_xlabel("Classification Threshold")
axes[0, 0].set_ylabel("Net Revenue Impact (₹)")
axes[0, 0].legend()
axes[0, 0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
)

# Chart 2: How many customers get flagged at each threshold
axes[0, 1].plot(
    results_df["threshold"], results_df["n_flagged"],
    color="#9C27B0", linewidth=2
)
axes[0, 1].axvline(
    best_row["threshold"], color="#F44336", linestyle="--",
    label=f"Optimal: {best_row['threshold']}"
)
axes[0, 1].set_title("Customers Flagged for Intervention vs Threshold")
axes[0, 1].set_xlabel("Classification Threshold")
axes[0, 1].set_ylabel("Number of customers to contact")
axes[0, 1].legend()

# Chart 3: Recall and Precision vs Threshold
axes[1, 0].plot(
    results_df["threshold"], results_df["recall"] * 100,
    color="#4CAF50", linewidth=2, label="Recall (% of churners caught)"
)
axes[1, 0].plot(
    results_df["threshold"], results_df["precision"] * 100,
    color="#F44336", linewidth=2, label="Precision (% of flags that were right)"
)
axes[1, 0].axvline(
    best_row["threshold"], color="gray", linestyle="--",
    alpha=0.7, label=f"Optimal: {best_row['threshold']}"
)
axes[1, 0].set_title("Recall vs Precision Tradeoff")
axes[1, 0].set_xlabel("Classification Threshold")
axes[1, 0].set_ylabel("Percentage (%)")
axes[1, 0].legend()

# Chart 4: Dollar-value confusion matrix at optimal threshold
_, tp, tn, fp, fn = calculate_revenue_impact(y_test, xgb_probs, best_row["threshold"])

# Create a 2x2 grid showing both counts and dollar values
matrix_counts  = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
matrix_dollars = np.array([
    [tn * COST_TRUE_NEGATIVE, fp * COST_FALSE_POSITIVE],
    [fn * COST_FALSE_NEGATIVE, tp * COST_TRUE_POSITIVE]
])

im = axes[1, 1].imshow(
    matrix_dollars,
    cmap="RdYlGn", vmin=-100000, vmax=100000
)

labels = [["True Negative\n(Correct: stays)", "False Positive\n(Wrong: loyal flagged)"],
          ["False Negative\n(Missed: churner)", "True Positive\n(Caught churner)"]]

for i in range(2):
    for j in range(2):
        axes[1, 1].text(
            j, i,
            f"{labels[i][j]}\n\nCount: {matrix_counts[i, j]}\n₹{matrix_dollars[i, j]:,.0f}",
            ha="center", va="center", fontsize=9, fontweight="bold"
        )

axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(["Predicted: Stays", "Predicted: Churns"])
axes[1, 1].set_yticklabels(["Actual: Stays", "Actual: Churns"])
axes[1, 1].set_title(f"Dollar-Value Confusion Matrix\n(threshold = {best_row['threshold']})")

plt.colorbar(im, ax=axes[1, 1], label="Revenue impact (₹)")
plt.tight_layout()
plt.savefig("outputs/step4_business_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 7. SAVE RESULTS FOR STREAMLIT APP
# --------------------------------------------------
results_df.to_csv("outputs/threshold_results.csv", index=False)

import pickle
with open("outputs/cost_analysis.pkl", "wb") as f:
    pickle.dump({
        "results_df": results_df,
        "best_threshold": best_row["threshold"],
        "clv": CUSTOMER_LIFETIME_VALUE,
        "offer_cost": RETENTION_OFFER_COST,
        "baseline_loss": baseline_loss
    }, f)

print("Results saved.")
print()
print("SUMMARY:")
print(f"  Without this model: company loses ₹{baseline_loss:,} per period")
print(f"  With this model at optimal threshold: net impact ₹{best_row['revenue_impact']:,.0f}")
print(f"  That's a ₹{best_row['revenue_impact'] + baseline_loss:,.0f} improvement over doing nothing")
print()
# print("Step 4 complete! Run step5_shap_explain.py next.")