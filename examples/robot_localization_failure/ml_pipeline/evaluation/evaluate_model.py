# ml_pipeline/evaluation/evaluate_model.py

import argparse
import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
)

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import apply_scaler, add_temporal_features


def load_model_package(model_dir: Path):
    """Load model, scaler, feature list, and optional metadata from a given directory."""
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    metadata_path = model_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, feature_cols, metadata


def main():

    parser = argparse.ArgumentParser(description="Evaluate trained ML model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Name of the model directory inside artifacts/models/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on positive class probability (default: 0.5)",
    )
    args = parser.parse_args()

    # ---------------------------
    # Resolve model directory
    # ---------------------------
    model_dirs = [d for d in MODELS.iterdir() if d.is_dir()]

    if not model_dirs:
        raise RuntimeError("❌ No model directories found in artifacts/models/")

    if args.model_dir is not None:
        candidate = MODELS / args.model_dir
        if not candidate.exists():
            print("❌ Requested model does not exist.")
            print("Available models:")
            for d in model_dirs:
                print(" -", d.name)
            return
        model_dir = candidate
    else:
        print("\nAvailable models:")
        for i, d in enumerate(model_dirs):
            print(f"[{i}] {d.name}")
        idx = input("Select a model index: ").strip()
        if not idx.isdigit() or int(idx) not in range(len(model_dirs)):
            raise ValueError("Invalid selection.")
        model_dir = model_dirs[int(idx)]

    print(f"\n📦 Using model: {model_dir.name}")

    # Load model/scaler/features/metadata
    model, scaler, feature_cols, metadata = load_model_package(model_dir)

    # ---------------------------
    # Load evaluation dataset
    # ---------------------------
    eval_path = DATASETS / "eval.parquet"
    print(f"Reading evaluation dataset: {eval_path}")

    df = pl.read_parquet(eval_path).drop_nulls()

    # If the model was trained with temporal features, add them here
    use_temporal = False
    if metadata is not None and metadata.get("temporal_features"):
        use_temporal = True

    if use_temporal:
        print("🔧 Model expects TEMPORAL features → adding them to eval dataset...")
        df = add_temporal_features(df)
    else:
        print("ℹ️ Model does NOT use temporal features.")

    # Check required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df["is_delocalized"].to_numpy()

    # Scale if needed
    X_scaled = apply_scaler(X, scaler)

    # ---------------------------
    # Predict with custom threshold
    # ---------------------------
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
        thr = args.threshold
        print(f"\nUsing decision threshold: {thr:.3f}")
        y_pred = (y_proba >= thr).astype(int)
    else:
        print("⚠️ Model has no predict_proba → falling back to predict(). Threshold ignored.")
        y_pred = model.predict(X_scaled)
        y_proba = None

    # ---------------------------
    # Metrics
    # ---------------------------
    print("\n=== Evaluation Metrics ===")
    print(classification_report(y_true, y_pred))

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = fbeta_score(y_true, y_pred, beta=1.0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5)

    cm = confusion_matrix(y_true, y_pred)

    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"F0.5 Score: {f05:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm)

    # ---------------------------
    # Save predictions
    # ---------------------------
    out_path = model_dir / "eval_results.parquet"
    df_out = df.with_columns([
        pl.Series("prediction", y_pred),
        pl.Series("probability", y_proba if y_proba is not None else [None] * len(y_pred)),
    ])
    df_out.write_parquet(out_path)

    print(f"\n✔ Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
