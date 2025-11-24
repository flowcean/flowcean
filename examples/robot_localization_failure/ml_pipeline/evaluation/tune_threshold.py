# ml_pipeline/evaluation/tune_threshold.py

import json
import joblib
import polars as pl
import numpy as np
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, fbeta_score

from ml_pipeline.utils.paths import DATASETS, MODELS
from ml_pipeline.utils.common import apply_scaler, add_temporal_features


def load_model_package(model_dir: Path):
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    metadata = None
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return model, scaler, feature_cols, metadata


def main():

    # ---------------------------
    # choose model
    # ---------------------------
    model_dirs = [d for d in MODELS.iterdir() if d.is_dir()]
    if not model_dirs:
        raise RuntimeError("No model directories under artifacts/models")

    print("\nAvailable models:")
    for i, d in enumerate(model_dirs):
        print(f"[{i}] {d.name}")
    idx = int(input("Select a model index: ").strip())
    model_dir = model_dirs[idx]
    print(f"\n📦 Using model: {model_dir.name}")

    model, scaler, feature_cols, metadata = load_model_package(model_dir)

    # ---------------------------
    # load eval data
    # ---------------------------
    eval_path = DATASETS / "eval.parquet"
    print(f"Reading evaluation dataset: {eval_path}")
    df = pl.read_parquet(eval_path).drop_nulls()

    use_temporal = metadata is not None and metadata.get("temporal_features", False)
    if use_temporal:
        print("🔧 Model expects TEMPORAL features → adding them to eval dataset...")
        df = add_temporal_features(df)
    else:
        print("ℹ️ Model does NOT use temporal features.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in eval dataset: {missing}")

    X = df.select(feature_cols).to_numpy()
    y_true = df["is_delocalized"].to_numpy()

    X_scaled = apply_scaler(X, scaler)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model has no predict_proba, cannot sweep thresholds.")

    y_proba = model.predict_proba(X_scaled)[:, 1]

    # ---------------------------
    # sweep thresholds
    # ---------------------------
    print("\n=== Threshold sweep (F0.5, F1) ===")
    print("thr\tprec\trec\tF1\tF0.5")

    best_f1 = -1.0
    best_f1_thr = None
    best_f05 = -1.0
    best_f05_thr = None

    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        print(f"{thr:.2f}\t{prec:.3f}\t{rec:.3f}\t{f1:.3f}\t{f05:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = thr
        if f05 > best_f05:
            best_f05 = f05
            best_f05_thr = thr

    print("\nBest F1  = {:.3f} at thr = {:.2f}".format(best_f1, best_f1_thr))
    print("Best F0.5= {:.3f} at thr = {:.2f}".format(best_f05, best_f05_thr))


if __name__ == "__main__":
    main()
