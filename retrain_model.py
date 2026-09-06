"""
Retrain the cyber breach prediction model on 16 features (reduced from 40).
Generates model/cyber_model.joblib and prints verified metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os

# -- Configuration --------------------------------------------------
DATA_PATH = "data/Enterprise_Cyber_Kill_Chain_Dataset.csv"
MODEL_PATH = "model/cyber_model.joblib"

KEEP_FEATURES = [
    "Attack_Stage", "Company_Size",  # categorical
    "Lateral_Movement", "Privilege_Escalation", "Persistence",
    "Credential_Stolen", "Data_Exfiltration_GB", "Data_Encrypted",
    "Phishing_Click", "Detection_Time_Min", "Response_Time_Min",
    "Firewall", "MFA", "EDR", "CVSS_Score", "Patch_Age_Days",
]
TARGET = "Attack_Success"

CATEGORICAL_FEATURES = ["Attack_Stage", "Company_Size"]
NUMERICAL_FEATURES = [f for f in KEEP_FEATURES if f not in CATEGORICAL_FEATURES]

LEAKAGE_COLS = [
    "Recovery_Cost_USD", "Records_Compromised", "Downtime_Hours",
    "Financial_Loss_USD", "Cyber_Risk_Score", "Risk_Level", "Incident_Severity",
]
DROP_COLS = ["Incident_ID", "Timestamp"]

# -- 1. Load data ---------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Raw shape: {df.shape}")

# -- 2. Clean -------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
print(f"  Dropped {before - len(df)} duplicate rows -> {len(df)} rows")

df = df.drop(columns=[c for c in LEAKAGE_COLS + DROP_COLS if c in df.columns])

if df["Detection_Time_Min"].isna().any():
    median_val = df["Detection_Time_Min"].median()
    df["Detection_Time_Min"] = df["Detection_Time_Min"].fillna(median_val)
    print(f"  Imputed Detection_Time_Min missing values with median={median_val:.1f}")

# -- 3. Select 16 features + target --------------------------------
X = df[KEEP_FEATURES].copy()
y = df[TARGET].copy()
print(f"  Feature matrix: {X.shape}, Target: {y.shape}")
print(f"  Class balance: {y.value_counts().to_dict()}")

# -- 4. Train/test split -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# -- 5. Preprocessing pipeline -------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_FEATURES),
    ],
    remainder="passthrough",
    sparse_threshold=0.3,
)

# -- 6. Logistic Regression baseline -------------------------------
print("\n-- Logistic Regression (baseline) --")
lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
print(f"  Accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
print(f"  F1 (breach): {f1_score(y_test, lr_pred):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, lr_proba):.4f}")

# -- 7. Random Forest (primary model) ------------------------------
print("\n-- Random Forest (primary) --")
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
    )),
])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)
rf_proba = rf_pipe.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)
auc = roc_auc_score(y_test, rf_proba)

print(f"  Accuracy:  {acc:.4f}")
print(f"  F1 (breach): {f1:.4f}")
print(f"  ROC-AUC:   {auc:.4f}")
print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
print(f"\n  Classification Report:\n{classification_report(y_test, rf_pred)}")

# -- 7b. Cross-validation for robustness ----------------------------
print("\n-- 5-Fold Cross-Validation --")
cv_scores = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# -- 8. Feature importance -----------------------------------------
print("\n-- Feature Importances (Random Forest) --")
rf_model = rf_pipe.named_steps["classifier"]
ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
all_feature_names = cat_feature_names + NUMERICAL_FEATURES
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in sorted_idx:
    print(f"  {all_feature_names[i]:30s} {importances[i]:.4f}")

# -- 9. Save model with compression --------------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf_pipe, MODEL_PATH, compress=3)
print(f"\nModel saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")

# -- 10. Smoke test ------------------------------------------------
print("\n-- Smoke test --")
loaded = joblib.load(MODEL_PATH)
test_row = pd.DataFrame([{
    "Attack_Stage": "Reconnaissance", "Company_Size": "Large",
    "Lateral_Movement": 0, "Privilege_Escalation": 0, "Persistence": 0,
    "Credential_Stolen": 0, "Data_Exfiltration_GB": 0.0, "Data_Encrypted": 0,
    "Phishing_Click": 0, "Detection_Time_Min": 25, "Response_Time_Min": 10,
    "Firewall": 1, "MFA": 1, "EDR": 1, "CVSS_Score": 3.0, "Patch_Age_Days": 10,
}])
prob = loaded.predict_proba(test_row)[0, 1]
print(f"  Case A (contained) breach probability: {prob:.4f}")

test_row2 = pd.DataFrame([{
    "Attack_Stage": "Impact", "Company_Size": "Small",
    "Lateral_Movement": 1, "Privilege_Escalation": 1, "Persistence": 1,
    "Credential_Stolen": 1, "Data_Exfiltration_GB": 85.0, "Data_Encrypted": 1,
    "Phishing_Click": 1, "Detection_Time_Min": 210, "Response_Time_Min": 150,
    "Firewall": 0, "MFA": 0, "EDR": 0, "CVSS_Score": 9.4, "Patch_Age_Days": 160,
}])
prob2 = loaded.predict_proba(test_row2)[0, 1]
print(f"  Case B (critical) breach probability: {prob2:.4f}")

print("\nDone.")
