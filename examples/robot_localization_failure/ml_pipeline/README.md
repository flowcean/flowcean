# 📘 **AMCL Failure Detection — ML Pipeline Documentation**

This repository provides a complete, modular, reproducible **machine learning pipeline** for detecting **AMCL delocalization** events using handcrafted features from:

* **Scan–Map Statistics**
* **Particle Cloud Statistics**
* **AMCL & Ground-truth pose alignment**

The pipeline:

1. **Processes ROS2 bags**
2. **Extracts 40+ handcrafted features**
3. **Builds train/eval datasets**
4. **Trains ML models**
5. **Evaluates any saved model**
6. **Stores all results cleanly inside `artifacts/`**
7. **Can run each step individually or automatically**

---

# 📂 **Directory Structure**

```
ml_pipeline/
│
├── dataset/
│   ├── build_dataset.py      # Process all bags → create training & eval datasets
│   ├── bag_processor.py      # Core logic for handling a single bag file
│   ├── helpers.py            # Common utilities (yaw, timeseries unpacking)
│
├── training/
│   ├── common.py             # Shared ML utilities (scaling, loading model package)
│   ├── train_model_rf.py     # Train RandomForest classifier + save artifacts
│
├── evaluation/
│   ├── evaluate_model.py     # Evaluate ANY model directory (interactive or CLI)
│
├── utils/
│   ├── paths.py              # Normalized paths for datasets, models, evaluation
│   ├── cleaning.py           # Deletes the entire artifacts directory
│
├── artifacts/                # Auto-created, contains datasets/models/eval outputs
│
├── run_all.py                # One-shot: dataset → training → evaluation
├── clean.py                  # Shortcut to wipe artifacts/
└── README.md                 # This file
```

---

# ⚙️ **Configuration File**

You must place your `config.yaml` at:

```
robot_localization_failure/config.yaml
```

Example:

```yaml
rosbag:
  training_paths:
    - toy_data/training_data_half/rec_20250923_135805_id_01
    - toy_data/training_data_half/rec_20250923_135805_id_02

  evaluation_paths:
    - toy_data/testing_data_half/rec_20250923_142005_id_01
    - toy_data/testing_data_half/rec_20250923_142005_id_02

  message_paths:
    - ros_msgs/sensor_msgs/msg/LaserScan.msg
    - ros_msgs/nav2_msgs/msg/Particle.msg
    - ros_msgs/nav2_msgs/msg/ParticleCloud.msg

localization:
  position_threshold: 0.4
  heading_threshold: 0.4
```

This file controls **which bags to process** and the **AMCL failure criteria**.

---

# 📦 **Artifacts Directory**

Everything the pipeline produces is saved inside:

```
ml_pipeline/artifacts/
```

Structure:

```
artifacts/
│
├── datasets/
│   ├── train.parquet
│   └── eval.parquet
│
├── models/
│   ├── random_forest_baseline/         # every training run creates its own folder
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_columns.json
│   ├── rf_experiment_02/
│   └── ...
│
└── evaluation/
    ├── model_name/
    │   ├── eval_results.parquet
    │   └── metrics.txt  (optional)
```

This ensures:

* every experiment is preserved
* model/eval artifacts never overwrite each other
* the repository root stays clean

---

# 🧠 **Overview of the Pipeline Components**

## 🔹 1. `build_dataset.py`

Processes all bags and produces:

```
artifacts/datasets/train.parquet
artifacts/datasets/eval.parquet
```

Each file contains:

* 15 Scan–Map features
* 14 Particle cloud statistics
* 7 AMCL pose quantities
* 7 GT pose quantities
* Position & heading errors
* `is_delocalized` label

A total of **40+ ML features**.

---

## 🔹 2. `train_model_rf.py`

Trains a baseline **Random Forest** classifier:

* loads `train.parquet`
* removes leakage columns
* scales features using `StandardScaler`
* trains RandomForest classifier
* saves everything to:

```
artifacts/models/<model_name>/
    model.pkl
    scaler.pkl
    feature_columns.json
```

Each training run creates a **new model directory**.

---

## 🔹 3. `common.py`

This file provides **shared ML utilities**:

✔ `load_model_package()`
Loads a stored model directory:

* model.pkl
* scaler.pkl
* feature_columns.json

✔ `apply_scaler()`
Applies a stored scaler or returns raw features if none exists.

✔ `timestamped_model_dir()`
Creates a new folder inside `artifacts/models/` automatically.

This is used by all training scripts to maintain clean, isolated experiment folders.

---

## 🔹 4. `evaluate_model.py`

A **model-agnostic evaluator**.

### 🟢 Supports:

### **1️⃣ Interactive model selection**

```
python3 -m ml_pipeline.evaluation.evaluate_model
```

Output:

```
Available models:
[0] random_forest_2025_11_20_14_33_51
[1] rf_experiment_bigtrees
Select a model index:
```

### **2️⃣ Direct model selection**

```
python3 -m ml_pipeline.evaluation.evaluate_model --model_dir random_forest_2025_11_20_14_33_51
```

### It performs:

* loads `eval.parquet`

* loads the chosen model

* scales features

* predicts labels + probabilities

* prints:

  ✔ classification report
  ✔ confusion matrix
  ✔ precision, recall, F1
  ✔ **F0.5 score (important for early detection)**

* saves predictions to:

```
artifacts/models/<model_dir>/eval_results.parquet
```

---

# 🚀 **How to Run the Pipeline**

## **1️⃣ Build the training & evaluation datasets**

Extracts all handcrafted features and builds 40+ column ML tables.

```
python3 -m ml_pipeline.dataset.build_dataset
```

This creates:

```
artifacts/datasets/train.parquet
artifacts/datasets/eval.parquet
```

---

## **2️⃣ Train a model**

Trains a Random Forest and saves artifacts.

```
python3 -m ml_pipeline.training.train_model_rf
```

This creates:

```
artifacts/models/random_forest_<timestamp>/
```

---

## **3️⃣ Evaluate a model**

### **Option A — interactive choose model**

```
python3 -m ml_pipeline.evaluation.evaluate_model
```

### **Option B — specify model name**

```
python3 -m ml_pipeline.evaluation.evaluate_model --model_dir random_forest_2025_11_20_14_33_51
```

Outputs include:

* precision, recall
* confusion matrix
* **F0.5 score**
* predictions saved to:

```
artifacts/models/<model_dir>/eval_results.parquet
```

---

# 🧹 **Cleaning Everything**

To wipe all datasets, models, and evaluation outputs:

```
python3 -m ml_pipeline.clean
```

This deletes the entire `artifacts/` directory.

---

# 🔁 **Run the Entire Pipeline Automatically**

```
python3 -m ml_pipeline.run_all
```

Runs:

```
build_dataset → train_model → evaluate_model
```

Everything is written into `artifacts/`.

---

# 🧬 **What the Pipeline Actually Computes**

### ✔ 1. Scan–Map Statistics

Geometric scan-map alignment → 15 features
(e.g., point_distance, line_angle, ray_quality…)

### ✔ 2. Particle Cloud Statistics

Distribution shape → 14 features
(center of gravity spread, cluster variance…)

### ✔ 3. Pose Alignment

Time-synced AMCL & ground truth
→ computed yaw, errors, labels

### ✔ 4. Final ML Table

A clean, flat dataset suitable for:

* Random Forest
* XGBoost
* LightGBM
* Neural networks
* Optuna hyperparameter tuning
* Scikit-learn pipelines
* Neural networks

---

# 🎯 **Next Steps (Planned Modules)**

You will soon add:

* `train_model_xgb.py` — XGBoost
* `train_model_lightgbm.py` — LightGBM
* `train_model_nn.py` — small MLP
* `evaluate_model.py` (extended) — ROC curves, PR curves
* `compare_models.py` — automatically compare all models
* `plot_feature_importance.py` — SHAP + RF/XGB importance

These will plug into the same `common.py` utilities.

---

# 🎉 **Conclusion**

This pipeline is:

* modular
* reproducible
* extensible
* model-agnostic
* easy to run
* easy to clean
