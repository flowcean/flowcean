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


def save_model(model_name: str, model, scaler, feature_cols, metrics):
    """
    Saves a model package into a UNIQUE timestamped folder:
    
    artifacts/models/<model_name>_<timestamp>/
        model.pkl
        scaler.pkl
        feature_columns.json
        metrics.json
    """

    # Create unique folder
    model_dir_name = f"{model_name}_{timestamp()}"
    model_dir = MODELS / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_dir / "model.pkl")

    # Save scaler (may be None)
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save features
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

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
