import json
import joblib
import polars as pl
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score
)

from ml_pipeline.utils.paths import DATASETS, MODELS, RESULTS


def main():

    model = joblib.load(MODELS / "model.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")

    with open(MODELS / "feature_columns.json") as f:
        feature_cols = json.load(f)

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

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:,1]

    print("\n=== Evaluation ===")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = fbeta_score(y_true, y_pred, beta=1.0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)

    print("\nMetrics:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("F0.5:", f05)

    out = df.with_columns([
        pl.Series("predicted", y_pred),
        pl.Series("probability", y_proba),
    ])
    out.write_parquet(RESULTS / "eval_results.parquet")

    print("✔ Saved eval_results.parquet")


if __name__ == "__main__":
    main()
