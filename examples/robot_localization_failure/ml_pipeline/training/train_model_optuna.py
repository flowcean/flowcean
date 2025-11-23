import numpy as np
import polars as pl
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score

import optuna

# Optional XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.utils.common import (
    load_dataset,
    prepare_features,
    add_temporal_features,
    compute_metrics,
    print_metrics,
    save_model,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

MODEL_NAME = "optuna"
USE_TEMPORAL_FEATURES = True
USE_SCANMAP_FEATURES = True

N_TRIALS = 50
RANDOM_STATE = 42
F_BETA = 0.5   # optimize F0.5


def create_model(model_type: str, trial: optuna.Trial, y_train):
    """
    Build a model for a given type and Optuna trial.
    """
    if model_type == "rf":
        n_estimators = trial.suggest_int("rf_n_estimators", 200, 800, step=100)
        max_depth = trial.suggest_int("rf_max_depth", 4, 20)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return model

    elif model_type == "xgb" and HAS_XGB:
        # basic imbalance handling
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = trial.suggest_float("xgb_scale_pos_weight", 1.0, max(1.0, neg / max(1, pos)))

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            n_estimators=trial.suggest_int("xgb_n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("xgb_max_depth", 4, 12),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
            gamma=trial.suggest_float("xgb_gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 0.0, 5.0),
            scale_pos_weight=scale_pos_weight,
        )
        return model

    else:
        raise ValueError(f"Unsupported or unavailable model_type: {model_type}")


def main():
    # ============================================================
    # 1) Load & prepare dataset
    # ============================================================
    print("Loading training data...")
    df = load_dataset(DATASETS / "train.parquet")

    if USE_TEMPORAL_FEATURES:
        print("Adding temporal features...")
        df = add_temporal_features(df)
    else:
        print("Skipping temporal features.")

    print("Preparing features...")
    X, y, feature_cols = prepare_features(df, use_scanmap_features=USE_SCANMAP_FEATURES)

    # Global train/val split used by ALL trials
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ============================================================
    # 2) Define Optuna objective
    # ============================================================
    def objective(trial: optuna.Trial) -> float:
        # Choose which model family
        model_types = ["rf"]
        if HAS_XGB:
            model_types.append("xgb")

        model_type = trial.suggest_categorical("model_type", model_types)

        model = create_model(model_type, trial, y_train)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # F0.5
        f05 = fbeta_score(y_val, y_pred, beta=F_BETA)

        # Optuna will maximize
        return f05

    # ============================================================
    # 3) Run study
    # ============================================================
    print(f"\n🚀 Starting Optuna study for {N_TRIALS} trials (optimize F{F_BETA})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n==============================")
    print("🎯 Optuna Best Trial")
    print("==============================")
    print("Best F0.5:", study.best_value)
    print("Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    best_params = study.best_params.copy()
    best_model_type = best_params.pop("model_type")

    # ============================================================
    # 4) Rebuild best model and train on FULL dataset
    # ============================================================
    print(f"\nRe-training best {best_model_type} on FULL dataset...")
    # Recreate model using best params
    def rebuild_model(model_type: str, params: dict):
        dummy_trial = optuna.trial.FixedTrial(params | {"model_type": model_type})
        return create_model(model_type, dummy_trial, y)

    best_model = rebuild_model(best_model_type, best_params)

    # Fit on full data (X, y)
    best_model.fit(X, y)

    y_pred_full = best_model.predict(X)
    metrics = compute_metrics(y, y_pred_full)

    print("\n=== Training-set metrics for best Optuna model ===")
    print_metrics(metrics)

    extra_metadata = {
        "optuna": True,
        "best_model_type": best_model_type,
        "best_params": study.best_params,
        "best_f05": float(study.best_value),
        "n_trials": N_TRIALS,
        "temporal_features": USE_TEMPORAL_FEATURES,
    }

    save_model(
        model_name=f"{MODEL_NAME}_{best_model_type}",
        model=best_model,
        scaler=None,  # tree models don't need external scaler
        feature_cols=feature_cols,
        metrics=metrics,
        extra_metadata=extra_metadata,
    )


if __name__ == "__main__":
    main()
