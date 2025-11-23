# ml_pipeline/training/train_model_xgb.py

import xgboost as xgb
from sklearn.model_selection import train_test_split

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    fit_scaler,
    apply_scaler,
    compute_metrics,
    print_metrics,
    save_model,
    add_temporal_features,
)

MODEL_NAME = "xgboost"          #xgboost_temporal
USE_TEMPORAL_FEATURES = False
USE_SCANMAP_FEATURES = False


def main():

    # ============================================================
    # 1) Load & prepare dataset
    # ============================================================
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    # ============================================================
    # 1.5) Optional temporal feature generation
    # ============================================================
    if USE_TEMPORAL_FEATURES:
        print("Adding temporal features...")
        df = add_temporal_features(df)
    else:
        print("Skipping temporal features.")

    print("Preparing features...")
    X, y, feature_cols = prepare_features(df, use_scanmap_features=USE_SCANMAP_FEATURES)

    # ============================================================
    # 2) Train/val split
    # ============================================================
    print("Splitting train/val...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y
    )

    # ============================================================
    # 3) Scaling? (NO — XGBoost performs better unscaled)
    # ============================================================
    scaler = fit_scaler(X_train, model_type="xgb")  # returns None
    X_train_scaled = apply_scaler(X_train, scaler)
    X_val_scaled = apply_scaler(X_val, scaler)

    # ============================================================
    # 4) Train model
    # ============================================================
    print(f"\nTraining {MODEL_NAME}...")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",  # good for CPU
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=(len(y) - sum(y)) / sum(y),   # class imbalance fix
        random_state=42,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

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
        metrics=metrics,
        extra_metadata={
            "temporal_features": USE_TEMPORAL_FEATURES,
            "model_type": MODEL_NAME,
            "notes": "trained using single map simulation data",
            "use_scanmap_features": USE_SCANMAP_FEATURES
        }
    )


if __name__ == "__main__":
    main()
