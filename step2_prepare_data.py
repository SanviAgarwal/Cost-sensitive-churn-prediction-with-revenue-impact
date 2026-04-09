# =============================================================
# STEP 2 — PREPARE THE DATA FOR MACHINE LEARNING
# =============================================================
# What this file does:
#   - Converts text columns to numbers (computers can't read "Yes"/"No")
#   - Handles the imbalance (only 26% churned)
#   - Splits data into training set and test set
#   - Saves the prepared data so Step 3 can use it
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 1. LOAD AND REPEAT THE BASIC FIXES FROM STEP 1
# --------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

print("=" * 50)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 50)
print(f"Starting with {len(df)} customers")
print()

# --------------------------------------------------
# 2. DROP COLUMNS WE DON'T NEED
# --------------------------------------------------
# customerID is just a unique ID — it has no predictive value
df.drop(columns=["customerID"], inplace=True)

# --------------------------------------------------
# 3. CONVERT THE TARGET COLUMN TO 0 AND 1
# --------------------------------------------------
# Our model needs numbers. "Yes" → 1 (churned), "No" → 0 (stayed)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print("Target column converted: 'Yes' → 1, 'No' → 0")
print(f"Churned (1): {df['Churn'].sum()}  |  Stayed (0): {(df['Churn']==0).sum()}")
print()

# --------------------------------------------------
# 4. CONVERT YES/NO TEXT COLUMNS TO 1/0
# --------------------------------------------------
# Many columns have "Yes" or "No" as values
yes_no_columns = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

for col in yes_no_columns:
    # Replace "No phone service" or "No internet service" → just treat as "No"
    df[col] = df[col].replace(
        {"No phone service": "No", "No internet service": "No"}
    )
    df[col] = (df[col] == "Yes").astype(int)

print("Converted all Yes/No columns to 1/0")

# --------------------------------------------------
# 5. CONVERT GENDER TO 0/1
# --------------------------------------------------
df["gender"] = (df["gender"] == "Male").astype(int)
print("Converted gender: Male → 1, Female → 0")

# --------------------------------------------------
# 6. CONVERT CONTRACT TYPE TO NUMBERS
# --------------------------------------------------
# Month-to-month is easiest to cancel = highest risk → 0
# One year = medium commitment → 1
# Two year = most committed → 2
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["Contract"] = df["Contract"].map(contract_map)
print("Contract: Month-to-month=0, One year=1, Two year=2")

# --------------------------------------------------
# 7. ONE-HOT ENCODE REMAINING TEXT COLUMNS
# --------------------------------------------------
# "One-hot encoding" means: if a column has 3 possible values,
# split it into 3 separate 0/1 columns.
# Example: InternetService = "DSL" becomes:
#   InternetService_DSL=1, InternetService_Fiber optic=0, InternetService_No=0

remaining_text_cols = ["InternetService", "PaymentMethod"]
df = pd.get_dummies(df, columns=remaining_text_cols, drop_first=False)
print(f"One-hot encoded: {remaining_text_cols}")
print()

# --------------------------------------------------
# 8. CREATE ONE NEW USEFUL FEATURE
# --------------------------------------------------
# How much does this customer pay per month they've been with us?
# High ratio = new customer paying a lot = higher churn risk
df["charge_per_month_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
print("New feature created: charge_per_month_tenure")
print()

# --------------------------------------------------
# 9. SEPARATE FEATURES (X) FROM TARGET (y)
# --------------------------------------------------
# X = everything the model learns from (all columns except Churn)
# y = what we want to predict (Churn column)
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"Features (X): {X.shape[1]} columns")
print(f"Target   (y): Churn (0 or 1)")
print()
print("Feature list:")
for col in X.columns:
    print(f"  - {col}")
print()

# --------------------------------------------------
# 10. SPLIT INTO TRAIN AND TEST
# --------------------------------------------------
# We keep 20% of data hidden as a "test set"
# The model NEVER sees this during training
# This is how we check if it actually learned anything real

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% goes to test set
    random_state=42,      # Fixed seed so results are reproducible
    stratify=y            # Keeps the 74%/26% split in both sets
)

print(f"Training set: {len(X_train)} customers (model learns from these)")
print(f"Test set:     {len(X_test)} customers  (model never sees these)")
print()

# --------------------------------------------------
# 11. SAVE EVERYTHING FOR THE NEXT STEPS
# --------------------------------------------------
with open("outputs/prepared_data.pkl", "wb") as f:
    pickle.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist()
    }, f)

# print("Prepared data saved to: outputs/prepared_data.pkl")
# print()
# print("NOTE ON CLASS IMBALANCE:")
# print("  Only 26% of customers churned. This is a problem because a lazy model")
# print("  could just say 'no one will churn' and be 74% accurate — but useless.")
# print("  In Step 3 we fix this using class_weight='balanced' in the model.")
# print()
# print("Step 2 complete! Run step3_train_model.py next.")