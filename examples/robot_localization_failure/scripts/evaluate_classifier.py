import polars as pl
import numpy as np
import json
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score,recall_score,fbeta_score

# ===========================
# Load model artifacts
# ===========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

print("Loaded feature columns:", feature_cols)

# ===========================
# Load parquet for inference
# ===========================
df = pl.read_parquet("eval.parquet")

# Safety: drop null rows
df = df.drop_nulls()

# Drop leakage if still present
drop_cols = [
    "time",
    "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
    "gt_yaw",
    "position_error", "heading_error_raw", "heading_error",
    "combined_error"
]
df = df.drop([c for c in drop_cols if c in df.columns])

# Ensure required columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in eval parquet: {missing}")

# Extract X and y
X = df.select(feature_cols).to_numpy()
y_true = df["is_delocalized"].to_numpy() if "is_delocalized" in df.columns else None

# Scale
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

# ===========================
# Evaluation
# ===========================
if y_true is not None:
    print("\n=== Evaluation Results ===")
    print(classification_report(y_true, y_pred))

    # Individual metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = fbeta_score(y_true, y_pred, beta=1.0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)

    print("\n=== Metric Summary ===")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"F0.5 Score:     {f05:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))
else:
    print("No ground-truth labels found. Outputting predictions only.")


# Save predictions
out = df.with_columns([
    pl.Series("predicted_is_delocalized", y_pred),
    pl.Series("deloc_probability", y_proba)
])

out.write_parquet("eval_results.parquet")

print("\nSaved predictions → eval_results.parquet")
