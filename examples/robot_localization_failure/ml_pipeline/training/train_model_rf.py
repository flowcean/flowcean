import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    fit_scaler,
    apply_scaler,
    compute_metrics,
    print_metrics,
    save_model,
)


MODEL_NAME = "random_forest"


def main():

    # ============================================================
    # 1) Load & prepare dataset
    # ============================================================
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    print("Preparing features...")
    X, y, feature_cols = prepare_features(df)

    # ============================================================
    # 2) Train/val split
    # ============================================================
    print("Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # ============================================================
    # 3) Scaling (RF does NOT need scaling → scaler = None)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="rf")

    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    # ============================================================
    # 4) Train model
    # ============================================================
    print(f"\nTraining {MODEL_NAME}...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ============================================================
    # 5) Evaluate
    # ============================================================
    y_pred = model.predict(X_val_scaled)

    print("\n=== Validation Performance ===")
    metrics = compute_metrics(y_val, y_pred)
    print_metrics(metrics)

    # ============================================================
    # 6) Save model package
    # ============================================================
    save_model(
        model_name=MODEL_NAME,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics
    )


if __name__ == "__main__":
    main()
