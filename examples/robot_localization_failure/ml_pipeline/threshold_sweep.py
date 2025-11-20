import json
import joblib
import numpy as np
import polars as pl

from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)

from ml_pipeline.utils.paths import DATASETS, MODELS, RESULTS


def main():

    # === Load model + preprocessing ===
    model = joblib.load(MODELS / "model.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")

    with open(MODELS / "feature_columns.json") as f:
        feature_cols = json.load(f)

    # === Load evaluation dataset ===
    df = pl.read_parquet(DATASETS / "eval.parquet").drop_nulls()

    drop_cols = [
        "time",
        "gt_x","gt_y","gt_qx","gt_qy","gt_qz","gt_qw",
        "gt_yaw",
        "position_error","heading_error_raw","heading_error",
        "combined_error"
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])

    X = df.select(feature_cols).to_numpy()
    y_true = df["is_delocalized"].to_numpy()

    X_scaled = scaler.transform(X)

    # === Predicted probabilities ===
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # === Threshold sweep ===
    thresholds = np.linspace(0, 1, 101)
    results = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results.append({
            "threshold": th,
            "precision": precision,
            "recall": recall,
            "F0.5": f05,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
        })

    # === Convert to DataFrame ===
    results_df = pl.DataFrame(results)

    # === Find best threshold by F0.5 ===
    best_row = results_df.sort("F0.5", descending=True).row(0)
    best_threshold = best_row[0]  # threshold is first column

    print("\n=== Best Threshold (F0.5) ===")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Precision: {best_row[1]:.3f}")
    print(f"Recall:    {best_row[2]:.3f}")
    print(f"F0.5:      {best_row[3]:.3f}")
    print(f"False Positives: {best_row[4]}")
    print(f"False Negatives: {best_row[5]}")
    print(f"True Positives:  {best_row[6]}")
    print(f"True Negatives:  {best_row[7]}")

    # === Save sweep results ===
    out_path = RESULTS / "threshold_sweep.parquet"
    results_df.write_parquet(out_path)

    print(f"\n✔ Saved full sweep to: {out_path}")
    print(results_df.head(10))


if __name__ == "__main__":
    main()
