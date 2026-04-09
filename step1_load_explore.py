# =============================================================
# STEP 1 — LOAD & EXPLORE THE DATA
# =============================================================
# What this file does:
#   - Loads the customer data from a CSV file
#   - Prints basic info so you understand what you're working with
#   - Makes 4 charts to visualise who churns and who doesn't
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 1. LOAD THE DATA
# --------------------------------------------------
# pd.read_csv reads a spreadsheet-style CSV file into Python
df = pd.read_csv("data/telco_churn.csv")

print("=" * 50)
print("DATASET LOADED")
print("=" * 50)
print(f"Rows (customers): {df.shape[0]}")
print(f"Columns (details per customer): {df.shape[1]}")
print()

# Show the first 5 rows so you can see what the data looks like
print("First 5 rows:")
print(df.head())
print()

# Show all column names
print("Columns available:")
print(df.columns.tolist())
print()

# --------------------------------------------------
# 2. FIX A KNOWN ISSUE IN THIS DATASET
# --------------------------------------------------
# TotalCharges column has some blank spaces instead of numbers
# We convert it to a number, and blanks become NaN (empty)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# How many missing values are there?
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

# Drop the 11 rows that have missing TotalCharges
df.dropna(subset=["TotalCharges"], inplace=True)
print(f"After dropping missing rows: {len(df)} customers remain")
print()

# --------------------------------------------------
# 3. UNDERSTAND THE TARGET: WHO CHURNED?
# --------------------------------------------------
# "Churn" is our target — the thing we want to predict
# It's a Yes/No column. Let's see the split.

churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100

print("Churn breakdown:")
print(f"  Stayed (No):  {churn_counts['No']}  ({churn_pct['No']:.1f}%)")
print(f"  Left   (Yes): {churn_counts['Yes']}  ({churn_pct['Yes']:.1f}%)")
print()
print("Note: Only ~26% of customers churned. This is called 'class imbalance'.")
print("      We'll handle this in Step 2.")
print()

# --------------------------------------------------
# 4. MAKE CHARTS
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Exploring Who Churns", fontsize=16, fontweight="bold")

# Chart 1: Simple pie chart of churn vs stay
axes[0, 0].pie(
    churn_counts,
    labels=["Stayed", "Churned"],
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90
)
axes[0, 0].set_title("Overall: How many customers left?")

# Chart 2: Tenure (how long they've been a customer) vs churn
# People who leave tend to leave early — this should show that
sns.histplot(
    data=df, x="tenure", hue="Churn",
    bins=30, ax=axes[0, 1],
    palette={"No": "#4CAF50", "Yes": "#F44336"}
)
axes[0, 1].set_title("Tenure: Long-term customers rarely leave")
axes[0, 1].set_xlabel("Months as a customer")
axes[0, 1].set_ylabel("Number of customers")

# Chart 3: Monthly Charges vs churn
# Higher bills = more likely to leave?
sns.boxplot(
    data=df, x="Churn", y="MonthlyCharges",
    ax=axes[1, 0],
    palette={"No": "#4CAF50", "Yes": "#F44336"}
)
axes[1, 0].set_title("Monthly Bill: Churners pay more on average")
axes[1, 0].set_xlabel("Did they churn?")
axes[1, 0].set_ylabel("Monthly bill (USD)")

# Chart 4: Contract type vs churn
# Month-to-month is easiest to cancel — should show highest churn
contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
).reset_index()
contract_churn.columns = ["Contract", "ChurnRate"]
sns.barplot(
    data=contract_churn, x="Contract", y="ChurnRate",
    ax=axes[1, 1],
    palette="Reds_r"
)
axes[1, 1].set_title("Contract Type: Month-to-month = most churn")
axes[1, 1].set_xlabel("Contract type")
axes[1, 1].set_ylabel("Churn rate (%)")

plt.tight_layout()
plt.savefig("outputs/step1_exploration.png", dpi=150, bbox_inches="tight")
plt.show()

# print("Charts saved to: outputs/step1_exploration.png")
# print()
# print("KEY TAKEAWAYS FROM CHARTS:")
# print("  1. Only 26% of customers churned — imbalanced dataset")
# print("  2. Customers who leave tend to leave within the first 12 months")
# print("  3. Churners pay higher monthly bills on average")
# print("  4. Month-to-month contract holders churn at ~43% — much higher than annual plans")
# print()
# print("Step 1 complete! Run step2_prepare_data.py next.")