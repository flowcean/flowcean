import json
import joblib
import numpy as np
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from ml_pipeline.utils.paths import DATASETS, MODELS


def main():

    df = pl.read_parquet(DATASETS / "train.parquet")
    df = df.drop_nulls()

    drop_cols = [
        "time",
        "gt_x","gt_y","gt_qx","gt_qy","gt_qz","gt_qw",
        "gt_yaw",
        "position_error","heading_error_raw","heading_error",
        "combined_error"
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])

    y = df["is_delocalized"].to_numpy()
    X = df.drop(["is_delocalized"])
    feature_cols = X.columns
    X = X.to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)

    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    joblib.dump(model, MODELS / "model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")

    with open(MODELS / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("✔ Saved model, scaler, feature list")


if __name__ == "__main__":
    main()
