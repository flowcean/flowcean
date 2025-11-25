import polars as pl
from ml_pipeline.utils.paths import DATASETS
import numpy as np

# Colors (optional, safe)
CY = "\033[96m"
YL = "\033[93m"
RD = "\033[91m"
GR = "\033[92m"
RS = "\033[0m"

# ---------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------

print(f"{CY}Loading datasets...{RS}")

train_path = DATASETS / "train.parquet"
eval_path  = DATASETS / "eval.parquet"

train = pl.read_parquet(train_path)
eval_df = pl.read_parquet(eval_path)

print(f"Train shape: {train.shape}")
print(f"Eval  shape: {eval_df.shape}")

# ---------------------------------------------------------------------
# 2. Label distribution
# ---------------------------------------------------------------------

print(f"\n{CY}=== LABEL DISTRIBUTION ==={RS}")

def label_dist(df, name):
    pos = df["is_delocalized"].sum()
    neg = df.height - pos
    print(f"{name}: True={int(pos)}, False={int(neg)}, Pos%={pos/df.height*100:.1f}%")

label_dist(train, "TRAIN")
label_dist(eval_df, "EVAL")

# ---------------------------------------------------------------------
# 3. Error stats
# ---------------------------------------------------------------------

print(f"\n{CY}=== POSITION & HEADING ERROR STATS ==={RS}")

def error_stats(df, name):
    print(f"\n{name} position_error describe:")
    print(df["position_error"].describe())

    print(f"\n{name} abs_heading_error describe:")
    print(df["heading_error"].abs().describe())

error_stats(train, "TRAIN")
error_stats(eval_df, "EVAL")

# ---------------------------------------------------------------------
# 4. Mean error by label
# ---------------------------------------------------------------------

print(f"\n{CY}=== MEAN ERROR PER CLASS ==={RS}")

def mean_err(df, name):
    mean_pos_false = df.filter(~pl.col("is_delocalized"))["position_error"].mean()
    mean_pos_true  = df.filter(pl.col("is_delocalized"))["position_error"].mean()
    mean_h_false   = df.filter(~pl.col("is_delocalized"))["heading_error"].abs().mean()
    mean_h_true    = df.filter(pl.col("is_delocalized"))["heading_error"].abs().mean()
    print(f"{name}:")
    print(f"  False: pos={mean_pos_false:.3f}, heading={mean_h_false:.3f}")
    print(f"  True : pos={mean_pos_true:.3f}, heading={mean_h_true:.3f}")

mean_err(train, "TRAIN")
mean_err(eval_df, "EVAL")

# ---------------------------------------------------------------------
# 5. Particle cloud feature comparison
# ---------------------------------------------------------------------

particle_cols = [
    "num_clusters",
    "cog_mean_dist",
    "cog_standard_deviation",
    "main_cluster_variance_x",
    "main_cluster_variance_y",
]

print(f"\n{CY}=== PARTICLE FEATURE GLOBAL STATS ==={RS}")

for col in particle_cols:
    print(f"\n{YL}{col}{RS}")
    print("TRAIN:")
    print(train[col].describe())
    print("EVAL:")
    print(eval_df[col].describe())

# ---------------------------------------------------------------------
# 6. Particle cloud conditional separation (loc vs deloc)
# ---------------------------------------------------------------------

print(f"\n{CY}=== PARTICLE FEATURE SEPARATION (False vs True) ==={RS}")

def separation(df, name):
    print(f"\n{name}:")
    for col in particle_cols:
        f = df.filter(~pl.col("is_delocalized"))[col].mean()
        t = df.filter(pl.col("is_delocalized"))[col].mean()
        print(f"  {col}: False={f:.4f}, True={t:.4f}, diff={t-f:.4f}")

separation(train, "TRAIN")
separation(eval_df, "EVAL")

# ---------------------------------------------------------------------
print(f"\n{GR}=== INSPECTION COMPLETE ==={RS}")
