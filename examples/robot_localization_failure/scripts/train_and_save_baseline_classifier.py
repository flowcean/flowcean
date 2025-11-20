import polars as pl
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Load parquet ===
df = pl.read_parquet("train.parquet")

# Remove rows with NaN (good safety check)
df = df.drop_nulls()

# ===============================
# Remove leakage columns
# ===============================
drop_cols = [
    "time",
    "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
    "gt_yaw",
    "position_error", "heading_error_raw", "heading_error",
    "combined_error"
]

existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(existing_drop_cols)

print(f"Dropping leaky columns: {existing_drop_cols}")

# Target
y = df["is_delocalized"].to_numpy()
X = df.drop(["is_delocalized"])

feature_cols = X.columns
print("Final feature columns:", feature_cols)

X = X.to_numpy()

# ==================================
# Train/val split
# ==================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ==================================
# Scaling (tree models don't need this, but good for general ML)
# ==================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==================================
# Train classifier
# ==================================
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train_scaled, y_train)

# ==================================
# Evaluation
# ==================================
y_pred = model.predict(X_val_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_val, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_val, y_pred))

# ==================================
# Save model + scaler + feature list
# ==================================
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("feature_columns.json", "w") as f:
    json.dump(feature_cols, f)

print("\nSaved model.pkl, scaler.pkl, and feature_columns.json")
