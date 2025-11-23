import json
import joblib
import numpy as np
import polars as pl

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)

from ml_pipeline.utils.paths import MODELS


# ============================================================
#                COLUMN DEFINITIONS
# ============================================================

LABEL_COL = "is_delocalized"

LEAKY_COLUMNS = [
    "time",
    "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
    "gt_yaw",
    "position_error", "heading_error_raw", "heading_error",
    "combined_error",
]

# all models must use columns EXCLUDING these
def remove_leaky_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drops columns that leak ground-truth or post-hoc errors."""
    cols_to_drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
    return df.drop(cols_to_drop)


# ============================================================
#                DATA LOADING & PREPARATION
# ============================================================

def load_dataset(parquet_path: Path) -> pl.DataFrame:
    """Reads a parquet and removes null rows."""
    df = pl.read_parquet(parquet_path)
    df = df.drop_nulls()
    return df


def prepare_features(df: pl.DataFrame):
    """
    Removes leakage columns, extracts feature matrix X and label y,
    ensures consistent column order.
    """
    df = remove_leaky_columns(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' missing in dataset.")

    y = df[LABEL_COL].to_numpy()
    X_df = df.drop([LABEL_COL])

    feature_cols = X_df.columns
    X = X_df.to_numpy()

    return X, y, feature_cols


# ============================================================
#                      SCALING
# ============================================================

def fit_scaler(X_train, model_type: str):
    """
    Returns a StandardScaler if the model requires scaling.
    Tree-based models (RF, XGB) do NOT require scaling, so return None.
    """
    from sklearn.preprocessing import StandardScaler

    if model_type in ["rf", "random_forest", "xgb", "xgboost", "tree"]:
        # No scaling needed
        return None

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X, scaler):
    """Apply scaler if not None, otherwise return unscaled X."""
    if scaler is None:
        return X
    return scaler.transform(X)


# ============================================================
#                     METRIC UTILITIES
# ============================================================

def compute_metrics(y_true, y_pred):
    """Compute structured performance metrics for saving."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "f0.5": float(fbeta_score(y_true,y_pred,beta=0.5)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics


def print_metrics(metrics):
    """Pretty-print metrics."""
    print("\n=== Evaluation Metrics ===")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1']:.4f}")
    print(f"F0.5 score: {metrics['f0.5']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(metrics["classification_report"])


# ============================================================
#                  MODEL SAVE / LOAD
# ============================================================

# ml_pipeline/utils/common.py

import json
import joblib
from datetime import datetime
from ml_pipeline.utils.paths import MODELS
from pathlib import Path


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_model(
    model_name: str,
    model,
    scaler,
    feature_cols,
    metrics,
    extra_metadata: dict | None = None,
):
    """
    Saves a complete model package into a UNIQUE timestamped folder:

    artifacts/models/<model_name>_<timestamp>/
        model.pkl
        scaler.pkl
        feature_columns.json
        metrics.json
        metadata.json (optional)

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "random_forest", "nn", "xgboost").
    model : object
        Trained model instance.
    scaler : object or None
        Optional scaler (e.g., StandardScaler). Saved only if not None.
    feature_cols : list[str]
        List of features used during training.
    metrics : dict
        Validation metrics (precision/recall/F1/etc.)
    extra_metadata : dict or None
        Any additional info such as:
            {"temporal_features": True,
             "model_type": "rf",
             "notes": "..."}
    """

    # -----------------------------------------
    # Create unique timestamped directory
    # -----------------------------------------
    model_dir_name = f"{model_name}_{timestamp()}"
    model_dir = MODELS / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Save model
    # -----------------------------------------
    joblib.dump(model, model_dir / "model.pkl")

    # Save scaler (if exists)
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save feature column list
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save validation metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------------------
    # Save extra metadata (if provided)
    # -----------------------------------------
    if extra_metadata is not None:
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(extra_metadata, f, indent=2)

    print(f"\n✔ Model saved in: {model_dir}")
    return model_dir

def load_model(model_name: str):
    """
    Loads: model, scaler (maybe None), and feature columns
    """
    model_dir = MODELS / model_name
    model = joblib.load(model_dir / "model.pkl")

    scaler_path = model_dir / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "feature_columns.json", "r") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


# ============================================================
#                      PREDICTION
# ============================================================

def predict_with_model(model, scaler, X):
    """Apply scaling if needed and run inference."""
    X_scaled = apply_scaler(X, scaler)
    return model.predict(X_scaled), model.predict_proba(X_scaled)[:, 1]

def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add simple temporal / rolling features to the dataset.

    For each numeric base feature (excluding time, labels, GT, and error columns)
    this adds:
      - <col>_diff1   : first difference (col(t) - col(t-1))
      - <col>_mean5   : rolling mean over window=5
      - <col>_std5    : rolling std  over window=5

    The frame is first sorted by 'time' to ensure temporal order.
    """
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in the dataset for temporal features.")

    # Sort by time to ensure proper temporal order
    df = df.sort("time")

    # Columns that must NOT be used as base features
    exclude_cols = {
        "time",
        "is_delocalized",
        "gt_x", "gt_y", "gt_qx", "gt_qy", "gt_qz", "gt_qw",
        "gt_yaw",
        "position_error", "heading_error_raw", "heading_error",
        "combined_error",
    }

    # Select numeric columns that are not excluded
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    schema = df.schema

    base_features: list[str] = [
        name
        for name, dtype in schema.items()
        if name not in exclude_cols and isinstance(dtype, numeric_types)
    ]

    print(f"[add_temporal_features] Base features ({len(base_features)}): {base_features}")

    # For each base feature, add diff1, mean5, std5
    new_cols: list[pl.Expr] = []
    for col in base_features:
        new_cols.extend([
            # First difference
            (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_diff1"),
            # Rolling mean over window=5 (min_periods=1 so early rows are still valid)
            pl.col(col)
            .rolling_mean(window_size=5, min_samples=1)
            .alias(f"{col}_mean5"),
            # Rolling std over window=5
            pl.col(col)
            .rolling_std(window_size=5, min_samples=1)
            .alias(f"{col}_std5"),
        ])

    if not new_cols:
        print("[add_temporal_features] No temporal features created (no suitable columns).")
        return df

    df = df.with_columns(new_cols)

    print(f"[add_temporal_features] Added {len(new_cols)} new temporal feature columns.")
    return df
