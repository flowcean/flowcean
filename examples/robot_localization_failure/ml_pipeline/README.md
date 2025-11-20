# 📘 **AMCL Failure Detection ML Pipeline**

This repository contains a complete end-to-end **machine learning pipeline** for detecting AMCL delocalization events using **Scan-Map Statistics**, **Particle Cloud Statistics**, and **robot pose**.
It processes ROS2 bag files, extracts handcrafted features, generates datasets, trains an ML classifier, evaluates it, and stores results in a structured `artifacts/` directory.

---

# 📂 **Directory Structure**

```
ml_pipeline/
│
├── dataset/
│   ├── build_dataset.py      # Process all bags → create training & eval datasets
│   ├── bag_processor.py      # Core logic for processing a single bag
│   ├── helpers.py            # Helpers (time-series unpacking, yaw, etc.)
│
├── training/
│   └── train_model.py        # Train classifier, save model, scaler, metadata
│
├── evaluation/
│   └── evaluate_model.py     # Evaluate saved model on evaluation dataset
│
├── utils/
│   ├── paths.py              # Artifact storage paths (datasets, models, results)
│   └── cleaning.py           # Removes artifacts directory
│
├── run_all.py                # Runs dataset → training → eval in one command
├── clean.py                  # Shortcut to wipe artifacts/
└── README.md                 # This file
```

---

# ⚙️ **Configuration**

You must place your `config.yaml` in:

```
robot_localization_failure/config.yaml
```

Example:

```yaml
rosbag:
  training_paths:
    - toy_data/training_data_half/rec_20250923_135805_id_01
    # - toy_data/training_data_half/rec_20250923_135805_id_02
  evaluation_paths:
    - toy_data/testing_data_half/rec_20250923_142005_id_01
    # - toy_data/testing_data_half/rec_20250923_142005_id_02
  message_paths:
    - ros_msgs/sensor_msgs/msg/LaserScan.msg
    - ros_msgs/nav2_msgs/msg/Particle.msg
    - ros_msgs/nav2_msgs/msg/ParticleCloud.msg
localization:
  position_threshold: 0.4
  heading_threshold: 0.4
architecture:
  image_size: 150
  width_meters: 15.0
learning:
  batch_size: 128
  learning_rate: 0.0001
  epochs: 2
  model_path: "models/robot_localization.model"
optuna:
  storage: "sqlite:///optuna.db"
```

---

# 📦 **Artifacts Folder**

All generated files are stored under:

```
ml_pipeline/artifacts/
```

Structure:

```
artifacts/
│
├── datasets/
│   ├── train.parquet
│   ├── eval.parquet
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.json
│
└── evaluation/
    ├── eval_results.parquet
    ├── metrics.txt
```

This keeps the project root clean and fully reproducible.

---

# 🚀 **How to Run Each Stage**

### **1️⃣ Build the Datasets**

Processes all bag files listed in `config.yaml` and generates:

* `artifacts/datasets/training_dataset.parquet`
* `artifacts/datasets/evaluation_dataset.parquet`

Run:

```bash
python3 -m ml_pipeline.dataset.build_dataset
```

---

### **2️⃣ Train the Model**

This loads the training parquet, cleans the data, trains a RandomForest classifier, and saves:

* `model.pkl`
* `scaler.pkl`
* `feature_columns.json`

Run:

```bash
python3 -m ml_pipeline.training.train_model
```

---

### **3️⃣ Evaluate the Model**

Loads model + scaler + evaluation parquet and produces:

* metrics summary
* confusion matrix
* `eval_results.parquet` with predictions + probabilities

Run:

```bash
python3 -m ml_pipeline.evaluation.evaluate_model
```

---

# 🧹 **Cleaning the Pipeline**

To completely delete all generated datasets, models, and evaluation files:

```bash
python3 -m ml_pipeline.clean
```

This wipes the entire `artifacts/` directory.

---

# 🔁 **Run ALL Stages Automatically**

Instead of running each step manually, you can run:

```bash
python3 -m ml_pipeline.run_all
```

This performs:

```
build_dataset → train_model → evaluate_model
```

in order, and prints a summary.

---

# 🧠 **What the Pipeline Actually Does (High-Level Overview)**

### **✔ Extracts features from ROS2 bags using two major transforms:**

* **ScanMapStatistics** → 15–29 handcrafted geometric features
* **ParticleCloudStatistics** → 1–14 particle cloud descriptors

### **✔ Aligns all time series**

Using scan timestamps as the reference timeline.

### **✔ Computes additional pose-related signals**

* AMCL yaw
* MoMo ground truth yaw
* position error
* heading error
* delocalization label (`is_delocalized`)

### **✔ Creates a clean ML-ready tabular dataset**

### **✔ Trains a classifier**

(RandomForest by default, but can be replaced later)

### **✔ Evaluates on held-out datasets**

---