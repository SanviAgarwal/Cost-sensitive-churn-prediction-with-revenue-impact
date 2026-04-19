# =============================================================
# STEP 6 — INTERACTIVE STREAMLIT DASHBOARD
# =============================================================
# Run this with:   streamlit run step6_app.py
# It opens a web page in your browser automatically.
#
# What this dashboard does:
#   - Lets you move a threshold slider
#   - Instantly shows how revenue impact changes
#   - Shows the Rupee-value confusion matrix
#   - Shows the 3 intervention tiers (high/medium/low risk)
#   - Shows SHAP feature importance
# =============================================================
import sys; sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import shap
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)

# --------------------------------------------------
# LOAD ALL SAVED DATA
# --------------------------------------------------
@st.cache_resource
def load_data():
    with open("outputs/best_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("outputs/cost_analysis.pkl", "rb") as f:
        cost_data = pickle.load(f)
    with open("outputs/shap_values.pkl", "rb") as f:
        shap_data = pickle.load(f)
    return model_data, cost_data, shap_data

model_data, cost_data, shap_data = load_data()

model          = model_data["model"]
X_test         = model_data["X_test"]
y_test         = model_data["y_test"]
xgb_probs      = model_data["xgb_probs"]
feature_names  = model_data["feature_names"]

results_df     = cost_data["results_df"]
CLV            = cost_data["clv"]
OFFER_COST     = cost_data["offer_cost"]
baseline_loss  = cost_data["baseline_loss"]

shap_values    = shap_data["shap_values"]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📉 Churn Prediction — Business Impact Dashboard")
st.markdown("""
This dashboard shows the **real Rupee impact** of your ML model.
Move the threshold slider to see how business outcomes change.
""")

st.divider()

# --------------------------------------------------
# SIDEBAR — COST ASSUMPTIONS
# --------------------------------------------------
st.sidebar.header("⚙️ Business Assumptions")
st.sidebar.markdown("Adjust these to match your real numbers:")

clv_input = st.sidebar.slider(
    "Customer Lifetime Value (₹)", 
    min_value=100, max_value=2000, value=CLV, step=50
)
offer_input = st.sidebar.slider(
    "Retention Offer Cost (₹)", 
    min_value=5, max_value=200, value=OFFER_COST, step=5
)

st.sidebar.divider()
st.sidebar.markdown(f"""
**What each outcome costs:**
- ✅ Caught churner (TP): **+₹{clv_input - offer_input}**
- ❌ Missed churner (FN): **-₹{clv_input}**
- ⚠️ Wasted offer (FP): **-₹{offer_input}**
- ✓  Correct stay (TN): **₹0**
""")

# --------------------------------------------------
# THRESHOLD SLIDER (MAIN CONTROL)
# --------------------------------------------------
st.header("🎛️ Threshold Selector")
st.markdown("""
The **threshold** is the probability cutoff for flagging a customer as "likely to churn."
- Lower threshold → flag more customers → catch more churners → more wasted offers too
- Higher threshold → flag fewer customers → miss more churners → fewer wasted offers
""")

threshold = st.slider(
    "Classification Threshold",
    min_value=0.10, max_value=0.90,
    value=float(cost_data["best_threshold"]),
    step=0.01,
    help="Move this left to catch more churners. Move right to be more conservative."
)

# --------------------------------------------------
# RECALCULATE AT CHOSEN THRESHOLD
# --------------------------------------------------
y_pred = (xgb_probs >= threshold).astype(int)
tp = int(((y_pred == 1) & (y_test == 1)).sum())
tn = int(((y_pred == 0) & (y_test == 0)).sum())
fp = int(((y_pred == 1) & (y_test == 0)).sum())
fn = int(((y_pred == 0) & (y_test == 1)).sum())

revenue = (
    tp * (clv_input - offer_input) +
    fp * (-offer_input) +
    fn * (-clv_input) +
    tn * 0
)
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
n_flagged = tp + fp

# --------------------------------------------------
# TOP METRICS ROW
# --------------------------------------------------
st.divider()
st.header("📊 Business Metrics at This Threshold")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Net Revenue Impact",
    f"₹{revenue:,.0f}",
    help="How much this model saves vs the baseline of doing nothing"
)
col2.metric(
    "Churners Caught",
    f"{tp} / {tp+fn}",
    f"{recall:.0%} recall",
    help="Of all customers who actually churned, how many did we flag?"
)
col3.metric(
    "Offers Wasted",
    f"{fp}",
    f"-₹{fp * offer_input:,}",
    delta_color="inverse",
    help="Loyal customers we wrongly flagged"
)
col4.metric(
    "Churners Missed",
    f"{fn}",
    f"-₹{fn * clv_input:,}",
    delta_color="inverse",
    help="Churners we failed to flag — most expensive mistake"
)
col5.metric(
    "Customers to Contact",
    f"{n_flagged}",
    f"{n_flagged/len(y_test):.0%} of test set",
    help="Total customers the retention team needs to reach out to"
)

# --------------------------------------------------
# Rupee-VALUE CONFUSION MATRIX
# --------------------------------------------------
st.divider()
st.header("💰 Rupee-Value Confusion Matrix")

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(7, 5))
    
    matrix_vals = np.array([[tn, fp], [fn, tp]])
    matrix_Rupees = np.array([
        [0, fp * (-offer_input)],
        [fn * (-clv_input), tp * (clv_input - offer_input)]
    ])
    
    cell_labels = [
        [f"TRUE NEGATIVE\nCorrect: stays\n\nCount: {tn}\n₹0", 
         f"FALSE POSITIVE\nWrong: loyal flagged\n\nCount: {fp}\n-₹{fp*offer_input:,}"],
        [f"FALSE NEGATIVE\nMissed churner\n\nCount: {fn}\n-₹{fn*clv_input:,}",
         f"TRUE POSITIVE\nCaught churner\n\nCount: {tp}\n+₹{tp*(clv_input-offer_input):,}"]
    ]
    
    colors = [["#C8E6C9", "#FFECB3"], ["#FFCDD2", "#B3E5FC"]]
    
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colors[i][j], ec="white", lw=2))
            ax.text(j + 0.5, 1.5 - i, cell_labels[i][j],
                    ha="center", va="center", fontsize=9, fontweight="bold")
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted: Stays", "Predicted: Churns"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual: Churns", "Actual: Stays"], fontsize=10)
    ax.set_title(f"At Threshold = {threshold}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with right:
    # Revenue vs threshold curve with current position marked
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Recalculate with user's cost inputs
    rev_curve = []
    for _, row in results_df.iterrows():
        t = row["threshold"]
        tp_r = row["tp"]; fp_r = row["fp"]; fn_r = row["fn"]
        r = tp_r*(clv_input-offer_input) + fp_r*(-offer_input) + fn_r*(-clv_input)
        rev_curve.append(r)
    
    ax.plot(results_df["threshold"], rev_curve, color="#2196F3", linewidth=2)
    ax.axvline(threshold, color="#F44336", linestyle="--", linewidth=2,
               label=f"Current: {threshold}")
    ax.scatter([threshold], [revenue], color="#F44336", s=80, zorder=5)
    ax.fill_between(results_df["threshold"], rev_curve, alpha=0.1, color="#2196F3")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net Revenue Impact (₹)")
    ax.set_title("Revenue Impact vs Threshold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --------------------------------------------------
# INTERVENTION TIERS
# --------------------------------------------------
st.divider()
st.header("🎯 Intervention Strategy — 3-Tier Segmentation")

st.markdown("""
Based on churn probability, customers are split into 3 tiers.
Each tier gets a different action to balance cost and impact.
""")

high_risk   = (xgb_probs >= 0.70)
medium_risk = (xgb_probs >= 0.40) & (xgb_probs < 0.70)
low_risk    = (xgb_probs < 0.40)

t1, t2, t3 = st.columns(3)

with t1:
    n = high_risk.sum()
    actual_churners_in_tier = y_test[high_risk].sum()
    st.error(f"🔴 HIGH RISK (p ≥ 0.70)\n\n"
             f"**{n} customers**\n\n"
             f"Actual churners in this group: **{actual_churners_in_tier}**\n\n"
             f"**Action:** Immediate personal call + loyalty offer")

with t2:
    n = medium_risk.sum()
    actual_churners_in_tier = y_test[medium_risk].sum()
    st.warning(f"🟡 MEDIUM RISK (0.40–0.70)\n\n"
               f"**{n} customers**\n\n"
               f"Actual churners in this group: **{actual_churners_in_tier}**\n\n"
               f"**Action:** Email nudge + feature highlight")

with t3:
    n = low_risk.sum()
    actual_churners_in_tier = y_test[low_risk].sum()
    st.success(f"🟢 LOW RISK (p < 0.40)\n\n"
               f"**{n} customers**\n\n"
               f"Actual churners in this group: **{actual_churners_in_tier}**\n\n"
               f"**Action:** Monitor only. No budget spent.")

# --------------------------------------------------
# SHAP FEATURE IMPORTANCE
# --------------------------------------------------
st.divider()
st.header("🔍 What Features Drive Churn?")

shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    max_display=12,
    show=False
)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.title("Top Features That Predict Churn (global average impact)", fontsize=12)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""
**How to read this chart:**
- Longer bar = feature has more impact on the prediction
- Features at the top are the most important for predicting churn
- Low tenure + high monthly charges + month-to-month contract are usually the top drivers
""")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Churn Prediction with Business Framing | "
    "Model: XGBoost | Dataset: IBM Telco Churn"
)